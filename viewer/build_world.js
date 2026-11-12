#!/usr/bin/env node
/* Build a native Minecraft Java 26.2 Anvil save from a bundle's blocks.npy. */

const fs = require('node:fs')
const path = require('node:path')
const zlib = require('node:zlib')
const crypto = require('node:crypto')

const nbt = require(process.env.PRISMARINE_NBT_PATH || 'prismarine-nbt')

const [bundleArg, templateArg, outputArg] = process.argv.slice(2)
if (!bundleArg || !templateArg || !outputArg) {
  throw new Error('usage: node build_world.js<bundle-dir> <void-template-world> <output-world>')
}

const bundleDir = path.resolve(bundleArg)
const templateWorld = path.resolve(templateArg)
const outputWorld = path.resolve(outputArg)
const blocksPath = path.join(bundleDir, 'blocks.npy')
const vocabPath = path.join(bundleDir, 'block_state_vocab.json')
const manifestPath = path.join(bundleDir, 'manifest.json')
const templateRegion = path.join(templateWorld, 'dimensions/minecraft/overworld/region/r.0.0.mca')

for (const required of [blocksPath, vocabPath, manifestPath, templateRegion, path.join(templateWorld, 'level.dat')]) {
  if (!fs.existsSync(required)) throw new Error(`missing required input: ${required}`)
}
if (fs.existsSync(outputWorld)) throw new Error(`refusing to overwrite output World: ${outputWorld}`)

function parseNpy(filePath) {
  const buffer = fs.readFileSync(filePath)
  if (buffer.subarray(0, 6).toString('latin1') !== '\x93NUMPY') throw new Error('invalid NPY magic')
  const major = buffer[6]
  let headerLength
  let headerOffset
  if (major === 1) {
    headerLength = buffer.readUInt16LE(8)
    headerOffset = 10
  } else if (major === 2 || major === 3) {
    headerLength = buffer.readUInt32LE(8)
    headerOffset = 12
  } else {
    throw new Error(`unsupported NPY version ${major}`)
  }
  const header = buffer.subarray(headerOffset, headerOffset + headerLength).toString('latin1')
  if (!header.includes("'descr': '<u2'") || !header.includes("'fortran_order': False")) {
    throw new Error(`expected little-endian C-order uint16 NPY, got ${header}`)
  }
  const match = header.match(/'shape':\s*\(([^)]*)\)/)
  if (!match) throw new Error('NPY header has no shape')
  const shape = match[1].split(',').map((value) => value.trim()).filter(Boolean).map(Number)
  if (shape.length !== 3 || shape.some((value) => !Number.isInteger(value) || value <= 0)) {
    throw new Error(`expected three-dimensional NPY, got ${shape}`)
  }
  const dataOffset = headerOffset + headerLength
  const count = shape.reduce((left, right) => left * right, 1)
  if (buffer.length !== dataOffset + count * 2) throw new Error('NPY payload length does not match shape')
  return { buffer, shape, dataOffset, count }
}

function tokenAt(array, y, z, x) {
  const [, sizeZ, sizeX] = array.shape
  const offset = array.dataOffset + ((y * sizeZ + z) * sizeX + x) * 2
  return array.buffer.readUInt16LE(offset)
}

function readChunkRaw(regionPath, tableIndex = 0) {
  const region = fs.readFileSync(regionPath)
  const sectorOffset = region.readUIntBE(tableIndex * 4, 3)
  const sectorCount = region[tableIndex * 4 + 3]
  if (!sectorOffset || !sectorCount) throw new Error(`missing template chunk ${tableIndex} in ${regionPath}`)
  const offset = sectorOffset * 4096
  const compressedLength = region.readUInt32BE(offset)
  const compression = region[offset + 4] & 0x7f
  const compressed = region.subarray(offset + 5, offset + 4 + compressedLength)
  if (compression === 1) return zlib.gunzipSync(compressed)
  if (compression === 2) return zlib.inflateSync(compressed)
  if (compression === 3) return compressed
  throw new Error(`unsupported template compression ${compression}`)
}

function paletteEntry(state) {
  const bracket = state.indexOf('[')
  if (bracket < 0) return { Name: { type: 'string', value: state } }
  if (!state.endsWith(']')) throw new Error(`invalid block state ${state}`)
  const properties = {}
  for (const item of state.slice(bracket + 1, -1).split(',')) {
    const equals = item.indexOf('=')
    if (equals <= 0) throw new Error(`invalid block property ${item} in ${state}`)
    properties[item.slice(0, equals)] = { type: 'string', value: item.slice(equals + 1) }
  }
  return {
    Name: { type: 'string', value: state.slice(0, bracket) },
    Properties: { type: 'compound', value: properties }
  }
}

function packValues(values, bits) {
  const valuesPerLong = Math.floor(64 / bits)
  const output = Array.from({ length: Math.ceil(values.length / valuesPerLong) }, () => 0n)
  for (let index = 0; index < values.length; index += 1) {
    const longIndex = Math.floor(index / valuesPerLong)
    const bitOffset = BigInt((index % valuesPerLong) * bits)
    output[longIndex] |= BigInt(values[index]) << bitOffset
  }
  return output.map((value) => BigInt.asIntN(64, value))
}

function makeBlockStates(array, chunkX, chunkZ, sectionY, states) {
  const palette = []
  const paletteIndex = new Map()
  const values = new Array(4096)
  const baseX = chunkX * 16
  const baseZ = chunkZ * 16
  const baseY = (sectionY + 4) * 16
  for (let localY = 0; localY < 16; localY += 1) {
    for (let localZ = 0; localZ < 16; localZ += 1) {
      for (let localX = 0; localX < 16; localX += 1) {
        const worldX = baseX + localX
        const worldZ = baseZ + localZ
        const token = baseY + localY < array.shape[0] &&
          worldX < array.shape[2] && worldZ < array.shape[1]
          ? tokenAt(array, baseY + localY, worldZ, worldX)
          : 0
        if (token >= states.length || states[token] == null) throw new Error(`unknown global state id ${token}`)
        if (!paletteIndex.has(token)) {
          paletteIndex.set(token, palette.length)
          palette.push(paletteEntry(states[token]))
        }
        const voxel = localY * 256 + localZ * 16 + localX
        values[voxel] = paletteIndex.get(token)
      }
    }
  }
  const value = {
    palette: { type: 'list', value: { type: 'compound', value: palette } }
  }
  if (palette.length > 1) {
    const bits = Math.max(4, Math.ceil(Math.log2(palette.length)))
    value.data = { type: 'longArray', value: packValues(values, bits) }
  }
  return { type: 'compound', value }
}

function makeHeightmaps(array, chunkX, chunkZ) {
  const heights = new Array(256).fill(0)
  for (let localZ = 0; localZ < 16; localZ += 1) {
    for (let localX = 0; localX < 16; localX += 1) {
      const worldX = chunkX * 16 + localX
      const worldZ = chunkZ * 16 + localZ
      if (worldX >= array.shape[2] || worldZ >= array.shape[1]) continue
      for (let y = array.shape[0] - 1; y >= 0; y -= 1) {
        if (tokenAt(array, y, worldZ, worldX) !== 0) {
          heights[localZ * 16 + localX] = y + 1
          break
        }
      }
    }
  }
  const packed = packValues(heights, 9)
  const value = {}
  for (const name of ['OCEAN_FLOOR', 'MOTION_BLOCKING_NO_LEAVES', 'MOTION_BLOCKING', 'WORLD_SURFACE']) {
    value[name] = { type: 'longArray', value: packed }
  }
  return { type: 'compound', value }
}

function emptyList() {
  return { type: 'list', value: { type: 'end', value: [] } }
}

function makeBlockEntities(array, states, chunkX, chunkZ) {
  const entities = []
  const simple = new Set(['chest','trapped_chest','ender_chest','furnace','blast_furnace','smoker','barrel','bell','decorated_pot','campfire','soul_campfire','brewing_stand','enchanting_table','beacon','hopper','dispenser','dropper'])
  for (let y=0;y<array.shape[0];y++) for(let z=0;z<16;z++) for(let x=0;x<16;x++) {
    const wx=chunkX*16+x, wz=chunkZ*16+z
    if(wx>=array.shape[2] || wz>=array.shape[1]) continue
    const token=tokenAt(array,y,wz,wx)
    if(token===0) continue
    const name=states[token].split('[')[0].replace('minecraft:','')
    let id = simple.has(name) ? name : null
    if(name==='trapped_chest') id='chest'
    if(name==='soul_campfire') id='campfire'
    if(name==='spawner') id='mob_spawner'
    if(name.endsWith('_bed')) id='bed'
    if(name.endsWith('_banner')) id='banner'
    if(name.endsWith('_hanging_sign')) id='hanging_sign'
    else if(name.endsWith('_sign')) id='sign'
    if(id) entities.push({id:{type:'string',value:'minecraft:'+id},x:{type:'int',value:wx},y:{type:'int',value:y-64},z:{type:'int',value:wz}})
  }
  return {type:'list',value:{type:'compound',value:entities}}
}

function makeChunkNbt(templateRaw, array, states, chunkX, chunkZ) {
  const root = nbt.parseUncompressed(templateRaw, 'big')
  const chunk = root.value
  chunk.xPos.value = chunkX
  chunk.zPos.value = chunkZ
  chunk.yPos.value = -4
  chunk.Status.value = 'minecraft:full'
  chunk.DataVersion.value = 4903
  chunk.isLightOn = { type: 'byte', value: 0 }
  chunk.block_entities = makeBlockEntities(array, states, chunkX, chunkZ)
  chunk.block_ticks = emptyList()
  chunk.fluid_ticks = emptyList()
  chunk.structures = {
    type: 'compound',
    value: {
      References: { type: 'compound', value: {} },
      starts: { type: 'compound', value: {} }
    }
  }
  chunk.Heightmaps = makeHeightmaps(array, chunkX, chunkZ)
  chunk.PostProcessing = {
    type: 'list',
    value: {
      type: 'list',
      value: Array.from({ length: 24 }, () => ({ type: 'end', value: [] }))
    }
  }
  const sections = chunk.sections.value.value
  const byY = new Map(sections.map((section) => [section.Y.value, section]))
  for (let sectionY = -4; sectionY <= 19; sectionY += 1) {
    const section = byY.get(sectionY)
    if (!section) throw new Error(`template chunk has no section Y=${sectionY}`)
    section.block_states = makeBlockStates(array, chunkX, chunkZ, sectionY, states)
    const tile = (manifest.samples || []).find(s => chunkX*16 >= s.previewX && chunkX*16 < s.previewX+(s.sizeX || 32) && chunkZ*16 >= s.previewZ && chunkZ*16 < s.previewZ+(s.sizeZ || 32))
    section.biomes = {type:'compound',value:{palette:{type:'list',value:{type:'string',value:[(tile && tile.biome) || manifest.biome || 'minecraft:plains']}}}}
    delete section.BlockLight
    delete section.SkyLight
  }
  return nbt.writeUncompressed(root, 'big')
}

function chunkRecord(raw) {
  const compressed = zlib.deflateSync(raw, { level: 6 })
  const record = Buffer.alloc(5 + compressed.length)
  record.writeUInt32BE(1 + compressed.length, 0)
  record[4] = 2
  compressed.copy(record, 5)
  return record
}

function writeRegion(regionPath, entries) {
  const header = Buffer.alloc(8192)
  const records = []
  let sectorOffset = 2
  const timestamp = Math.floor(Date.now() / 1000)
  for (const entry of entries.sort((a, b) => a.index - b.index)) {
    const sectors = Math.ceil(entry.record.length / 4096)
    if (sectors > 255) throw new Error(`chunk ${entry.chunkX},${entry.chunkZ} needs ${sectors} sectors`)
    header.writeUIntBE(sectorOffset, entry.index * 4, 3)
    header[entry.index * 4 + 3] = sectors
    header.writeUInt32BE(timestamp, 4096 + entry.index * 4)
    const padded = Buffer.alloc(sectors * 4096)
    entry.record.copy(padded)
    records.push(padded)
    sectorOffset += sectors
  }
  fs.writeFileSync(regionPath, Buffer.concat([header, ...records]))
}

function copyWorldTemplate() {
  fs.cpSync(templateWorld, outputWorld, {
    recursive: true,
    filter(source) {
      const relative = path.relative(templateWorld, source)
      if (relative === 'session.lock' || relative === 'level.dat_old' || ['players','playerdata','advancements','stats'].some(p => relative === p || relative.startsWith(p + path.sep))) return false
      return !relative.startsWith('dimensions/minecraft/overworld/region') &&
        !relative.startsWith('dimensions/minecraft/overworld/entities') &&
        !relative.startsWith('dimensions/minecraft/overworld/poi')
    }
  })
  fs.mkdirSync(path.join(outputWorld, 'dimensions/minecraft/overworld/region'), { recursive: true })
  fs.mkdirSync(path.join(outputWorld, 'dimensions/minecraft/overworld/entities'), { recursive: true })
  fs.mkdirSync(path.join(outputWorld, 'dimensions/minecraft/overworld/poi'), { recursive: true })
}

function updateLevelDat() {
  const levelPath = path.join(outputWorld, 'level.dat')
  const level = nbt.parseUncompressed(zlib.gunzipSync(fs.readFileSync(levelPath)), 'big')
  const data = level.value.Data.value
  const now = BigInt(Date.now())
  data.LevelName.value = path.basename(outputWorld)
  data.LastPlayed.value = [
    Number(BigInt.asIntN(32, now >> 32n)),
    Number(BigInt.asIntN(32, now))
  ]
  data.GameType.value = 3
  delete data.Player
  delete data.player
  if (data.DayTime) data.DayTime.value = [0, 6000]
  if (data.Time) data.Time.value = [0, 6000]
  data.allowCommands.value = 1
  data.spawn.value.pos.value = manifest.spawn_xyz || [array.shape[2]/2, array.shape[0]-32, array.shape[1]/2]
  data.spawn.value.dimension.value = 'minecraft:overworld'
  data.spawn.value.yaw.value = 0
  data.spawn.value.pitch.value = 90
  fs.writeFileSync(levelPath, zlib.gzipSync(nbt.writeUncompressed(level, 'big')))
  const rulesPath=path.join(outputWorld,'data/minecraft/game_rules.dat')
  if(fs.existsSync(rulesPath)) {
    const rules=nbt.parseUncompressed(zlib.gunzipSync(fs.readFileSync(rulesPath)),'big')
    const entries=rules.value.data.value
    entries['minecraft:random_tick_speed']={type:'int',value:0}
    for(const key of ['spawn_mobs','spawn_monsters','spawner_blocks_work','advance_weather','advance_time','spread_vines']) entries['minecraft:'+key]={type:'byte',value:0}
    fs.writeFileSync(rulesPath,zlib.gzipSync(nbt.writeUncompressed(rules,'big')))
  }

}

const array = parseNpy(blocksPath)
if (![256, 384].includes(array.shape[0])) throw new Error(`expected Y=256 or Y=384, got ${array.shape}`)
const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'))
const acceptedFormats = new Set([
  'minecraft_java_26_2_unified_preview_v1',
  'minecraft_java_26_2_biome_examples_preview_v1',
  'minecraft_java_26_2_dataset_random_preview_v1',
  'minecraft_java_26_2_vae_gallery_v1',
  'minecraft_java_26_2_flow_validation_gallery_v1',
  'minecraft_java_26_2_half_flow_validation_gallery_v1',
  'minecraft_java_26_2_flow_unconditional_gallery_v1',
  'minecraft_java_26_2_crop_vae_validation_grid_v1',
  'minecraft_java_26_2_crop_vae_interpolation_grid_v1',
  'minecraft_java_26_2_crop_vae_sample_interpolation_gt_grid_v1',
  'minecraft_java_26_2_half_mask_diffusion_gallery_v1',
  'minecraft_java_26_2_half_mask_trace_gallery_v1'
])
if (!acceptedFormats.has(manifest.format) || manifest.data_version !== 4903) {
  throw new Error('unexpected Java gallery manifest')
}
if (JSON.stringify(manifest.shape_yzx) !== JSON.stringify(array.shape)) throw new Error('manifest/NPY shape mismatch')
const vocabulary = JSON.parse(fs.readFileSync(vocabPath, 'utf8'))
const states = vocabulary.idToState
if (!Array.isArray(states) || states[0] !== 'minecraft:air') throw new Error('bad Java block-state vocabulary')
const bundledDefaults = path.join(bundleDir, 'block_default_states.json')
const defaultsPath = fs.existsSync(bundledDefaults) ? bundledDefaults : path.join(__dirname, '../streamvoxel/common/block_defaults.json')
const defaults = JSON.parse(fs.readFileSync(defaultsPath, 'utf8'))
if (crypto.createHash('sha256').update(JSON.stringify(states)).digest('hex') !== defaults.registryStatesSha256) throw new Error('preview registry/defaults mismatch')
const normalized = states.map(s => defaults.blocks[s.split('[')[0]].rawStateId)
let normalizedCount = 0
for(let i=0;i<array.count;i++) {
  const offset=array.dataOffset+i*2, before=array.buffer.readUInt16LE(offset)
  if(before>=normalized.length) throw new Error('unknown raw preview state ID')
  const after=normalized[before]
  if(before!==after) normalizedCount++
  array.buffer.writeUInt16LE(after,offset)
}


const templateRaw = readChunkRaw(templateRegion)
const chunkCountX = Math.ceil(array.shape[2] / 16)
const chunkCountZ = Math.ceil(array.shape[1] / 16)
const regions = new Map()
let storedChunks = 0
const chunkBoxes = manifest.chunk_boxes_xz
if (chunkBoxes && (!Array.isArray(chunkBoxes) || chunkBoxes.some(b => !Array.isArray(b) || b.length !== 4 || b.some(v => !Number.isInteger(v)) || b[0] < 0 || b[1] < 0 || b[2] <= b[0] || b[3] <= b[1] || b[2] > chunkCountX || b[3] > chunkCountZ))) throw new Error('invalid sparse chunk boxes')
for (let chunkZ = 0; chunkZ < chunkCountZ; chunkZ += 1) {
  for (let chunkX = 0; chunkX < chunkCountX; chunkX += 1) {
    if (chunkBoxes && !chunkBoxes.some(b => chunkX >= b[0] && chunkX < b[2] && chunkZ >= b[1] && chunkZ < b[3])) continue
    const regionX = Math.floor(chunkX / 32)
    const regionZ = Math.floor(chunkZ / 32)
    const key = `${regionX},${regionZ}`
    if (!regions.has(key)) regions.set(key, [])
    const localX = chunkX - regionX * 32
    const localZ = chunkZ - regionZ * 32
    const raw = makeChunkNbt(templateRaw, array, states, chunkX, chunkZ)
    regions.get(key).push({
      index: localZ * 32 + localX,
      chunkX,
      chunkZ,
      record: chunkRecord(raw)
    })
    storedChunks += 1
  }
  console.log(`encoded chunk row ${chunkZ + 1}/${chunkCountZ}`)
}

copyWorldTemplate()
const regionDir = path.join(outputWorld, 'dimensions/minecraft/overworld/region')
for (const [key, entries] of regions) {
  const [regionX, regionZ] = key.split(',').map(Number)
  const regionPath = path.join(regionDir, `r.${regionX}.${regionZ}.mca`)
  writeRegion(regionPath, entries)
  console.log(`wrote ${path.basename(regionPath)} with ${entries.length} chunks`)
}
updateLevelDat()
fs.copyFileSync(manifestPath, path.join(outputWorld, 'DATA_PREVIEW_MANIFEST.json'))
const verification = {
  format: 'minecraft_java_26_2_preview_world_v1',
  statePolicy: 'java26_engine_default_v1',
  normalizedInputVoxels: normalizedCount,
  dataVersion: 4903,
  shapeYZX: array.shape,
  chunkGridXZ: [chunkCountX, chunkCountZ],
  storedChunks,
  regionFiles: [...regions.keys()].length,
  spawnXYZ: manifest.spawn_xyz || [array.shape[2]/2, array.shape[0]-32, array.shape[1]/2],
  layout: manifest.layout,
  sourceBundle: bundleDir
}
fs.writeFileSync(path.join(outputWorld, 'DATA_PREVIEW_VERIFICATION.json'), `${JSON.stringify(verification, null, 2)}\n`)
console.log(JSON.stringify({ outputWorld, ...verification }, null, 2))
