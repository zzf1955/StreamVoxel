from viewer.playback import PlayBuffer


def packet(seq, at):
    return dict(seq=seq, kind='frame', emitted_unix_ns=int(at * 1e9))


def test_waits_for_delay_then_plays_at_emit_pace():
    buffer = PlayBuffer(delay=2., max_gap=10.)
    for index in range(40):
        buffer.push(packet(index, 100 + index * .1), now=0.)
    assert [p['seq'] for p in buffer.due(0.)] == [0]          # 3.9 s buffered: starts at the first packet
    out = buffer.due(1.)
    assert [p['seq'] for p in out] == list(range(1, 11))      # playhead 100.0 -> 101.0


def test_short_buffer_starts_after_waiting():
    buffer = PlayBuffer(delay=2.)
    buffer.push(packet(0, 5.), now=0.)
    assert buffer.due(1.) == []
    assert [p['seq'] for p in buffer.due(2.5)] == [0]


def test_idle_gap_is_skipped_and_stall_counted():
    buffer = PlayBuffer(delay=0., max_gap=1.)
    buffer.push(packet(0, 10.), now=0.)
    assert [p['seq'] for p in buffer.due(0.)] == [0]
    assert buffer.due(0.1) == [] and not buffer.playing and buffer.stalls == 1
    buffer.push(packet(1, 500.), now=5.)
    assert [p['seq'] for p in buffer.due(5.)] == [1]


def test_slow_display_releases_several_packets_at_once():
    buffer = PlayBuffer(delay=0., max_gap=10.)
    for index in range(10):
        buffer.push(packet(index, 1 + index * .1), now=0.)
    buffer.due(0.)
    assert len(buffer.due(0.5)) >= 5
