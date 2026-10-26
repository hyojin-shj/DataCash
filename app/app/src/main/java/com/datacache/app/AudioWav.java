package com.datacache.app;

import android.annotation.SuppressLint;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.AudioTimestamp;
import android.media.MediaRecorder;

import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.RandomAccessFile;

/**
 * 폰 마이크 48kHz 모노 16비트 WAV. 첫 샘플의 BOOTTIME 시각을 AudioTimestamp 로 구해 audio.json 에 적는다
 * (음향 비콘 18~20kHz 를 들으려면 48kHz 가 필요하다).
 */
final class AudioWav {
    static final int RATE = 48000;
    private AudioRecord rec;
    private Thread thread;
    private volatile boolean running;
    private long startNs = -1, frames;
    private File out;

    @SuppressLint("MissingPermission")
    void start(File dir) {
        int min = AudioRecord.getMinBufferSize(RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
        rec = new AudioRecord(MediaRecorder.AudioSource.UNPROCESSED, RATE, AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT, Math.max(min, RATE / 5));
        if (rec.getState() != AudioRecord.STATE_INITIALIZED) {   // UNPROCESSED 를 못 쓰는 기기
            rec.release();
            rec = new AudioRecord(MediaRecorder.AudioSource.MIC, RATE, AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT, Math.max(min, RATE / 5));
        }
        out = new File(dir, "audio.wav");
        running = true;
        rec.startRecording();
        thread = new Thread(() -> {
            byte[] buf = new byte[RATE / 50 * 2];
            AudioTimestamp ts = new AudioTimestamp();
            try (FileOutputStream f = new FileOutputStream(out)) {
                f.write(new byte[44]);
                while (running) {
                    int n = rec.read(buf, 0, buf.length);
                    if (n > 0) {
                        f.write(buf, 0, n);
                        frames += n / 2;
                    }
                    if (startNs < 0 && rec.getTimestamp(ts, AudioTimestamp.TIMEBASE_BOOTTIME) == AudioRecord.SUCCESS) {
                        startNs = ts.nanoTime - ts.framePosition * 1_000_000_000L / RATE;   // 0번 샘플의 시각
                    }
                }
            } catch (Exception ignored) {
            }
        }, "audio");
        thread.start();
    }

    JSONObject stop(File dir) throws Exception {
        running = false;
        thread.join(2000);
        rec.stop();
        rec.release();
        writeHeader(out, frames);
        JSONObject info = new JSONObject().put("start_ns", startNs > 0 ? startNs : Clock.now()).put("rate", RATE)
                .put("start_ns_source", startNs > 0 ? "AudioTimestamp BOOTTIME" : "fallback");
        Files.write(new File(dir, "audio.json"), info.toString());
        return info;
    }

    private static void writeHeader(File f, long frames) throws Exception {
        long data = frames * 2;
        try (RandomAccessFile r = new RandomAccessFile(f, "rw")) {
            r.seek(0);
            r.writeBytes("RIFF");
            r.write(le32(36 + data));
            r.writeBytes("WAVEfmt ");
            r.write(le32(16));
            r.write(new byte[]{1, 0, 1, 0});
            r.write(le32(RATE));
            r.write(le32(RATE * 2L));
            r.write(new byte[]{2, 0, 16, 0});
            r.writeBytes("data");
            r.write(le32(data));
        }
    }

    private static byte[] le32(long v) {
        return new byte[]{(byte) v, (byte) (v >> 8), (byte) (v >> 16), (byte) (v >> 24)};
    }
}
