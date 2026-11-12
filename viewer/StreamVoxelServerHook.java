import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Launches the vanilla server bundler, keeps every player's flying speed at the value in a file, and drops
 * placed live templates from the server's template cache, which otherwise keeps every one of them forever
 * (each bridge row is a new template id; 3042 of them exhausted a 2 GB heap).
 *
 * Vanilla has no command for flySpeed. The server is reached through the "Server thread" Runnable, which
 * captures an AtomicReference to it (MinecraftServer.spin); everything else is public API called by
 * reflection, so nothing is compiled against Minecraft. Needs --add-opens java.base/java.lang=ALL-UNNAMED.
 * The bridge writes the file (absolute flySpeed, vanilla 0.05); a changed ability is sent to the client with
 * onUpdateAbilities, so an unmodified client follows it.
 */
public final class StreamVoxelServerHook {
    public static void main(String[] args) throws Throwable {
        Path file = Path.of(System.getProperty("streamvoxel.flySpeedFile", "fly_speed.txt"));
        Thread watcher = new Thread(() -> watch(file), "streamvoxel-server-hook");
        watcher.setDaemon(true);
        watcher.start();
        Class.forName("net.minecraft.bundler.Main").getMethod("main", String[].class).invoke(null, (Object) args);
    }

    static Object server() throws ReflectiveOperationException {
        for (Thread thread : Thread.getAllStackTraces().keySet()) {
            if (!"Server thread".equals(thread.getName())) {
                continue;
            }
            Field holder = Thread.class.getDeclaredField("holder");
            holder.setAccessible(true);
            Object fields = holder.get(thread);
            Field task = fields.getClass().getDeclaredField("task");
            task.setAccessible(true);
            Object runnable = task.get(fields);
            for (Field field : runnable.getClass().getDeclaredFields()) {
                field.setAccessible(true);
                if (field.get(runnable) instanceof AtomicReference<?> ref && ref.get() != null) {
                    return ref.get();
                }
            }
        }
        return null;
    }

    static void watch(Path file) {
        Object server = null;
        String failed = null;
        while (true) {
            try {
                Thread.sleep(500);
                if (server == null && (server = server()) != null) {
                    System.out.println("[streamvoxel] server hook attached to " + server.getClass().getName());
                }
                if (server == null) {
                    continue;
                }
                Object target = server;
                Method execute = target.getClass().getMethod("execute", Runnable.class);
                execute.invoke(target, (Runnable) () -> forgetTemplates(target));
                if (!Files.exists(file)) {
                    continue;
                }
                float speed = Float.parseFloat(Files.readString(file).trim());
                execute.invoke(target, (Runnable) () -> apply(target, speed));
            } catch (Throwable error) {
                if (!String.valueOf(error).equals(failed)) {
                    failed = String.valueOf(error);
                    System.out.println("[streamvoxel] fly speed hook: " + failed);
                }
            }
        }
    }

    /** Runs on the server thread, so every `place template` the bridge sent before it has finished. */
    static void forgetTemplates(Object server) {
        try {
            Object manager = server.getClass().getMethod("getStructureManager").invoke(server);
            Field repository = manager.getClass().getDeclaredField("structureRepository");
            repository.setAccessible(true);
            Map<?, ?> cache = (Map<?, ?>) repository.get(manager);
            List<Object> live = new ArrayList<>();
            for (Object id : cache.keySet()) {
                if (id.toString().startsWith("streamvoxel_live:")) {
                    live.add(id);
                }
            }
            for (Object id : live) {
                cache.remove(id);
            }
        } catch (ReflectiveOperationException error) {
            System.out.println("[streamvoxel] template cache: " + error);
        }
    }

    static void apply(Object server, float speed) {
        try {
            Object list = server.getClass().getMethod("getPlayerList").invoke(server);
            for (Object player : (List<?>) list.getClass().getMethod("getPlayers").invoke(list)) {
                Object abilities = player.getClass().getMethod("getAbilities").invoke(player);
                if ((Float) abilities.getClass().getMethod("getFlyingSpeed").invoke(abilities) != speed) {
                    abilities.getClass().getMethod("setFlyingSpeed", float.class).invoke(abilities, speed);
                    player.getClass().getMethod("onUpdateAbilities").invoke(player);
                }
            }
        } catch (ReflectiveOperationException error) {
            System.out.println("[streamvoxel] fly speed apply: " + error);
        }
    }
}
