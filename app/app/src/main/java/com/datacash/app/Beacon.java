package com.datacash.app;

import android.media.AudioAttributes;
import android.media.AudioFormat;
import android.media.AudioTrack;

import java.io.File;
import java.io.FileWriter;

/**
 * 음향 비콘: 2초마다 18→20kHz 0.05초 처프 (대부분의 사람 귀에 안 들림). 낸다고 명령한 BOOTTIME 시각을 beacon.csv 에.
 * 폰 마이크와 워치 마이크가 같이 듣고, 서버가 두 시계를 ms 이하로 맞춘다 (손목 톡톡 같은 사용자 동작이 필요 없다).
 */
final class Beacon {
    static final int RATE = 48000;
    static final double F0 = 18000, F1 = 20000, DUR = 0.05, PERIOD = 2.0;
    private AudioTrack track;
    private Thread thread;
    private volatile boolean running;
    private FileWriter log;

    void start(File dir) throws Exception {
        int n = (int) (RATE * DUR);
        short[] pcm = new short[n];
        double k = (F1 - F0) / DUR;
        for (int i = 0; i < n; i++) {
            double t = i / (double) RATE;
            double w = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (n - 1));
            pcm[i] = (short) (Math.sin(2 * Math.PI * (F0 * t + 0.5 * k * t * t)) * w * 0.35 * Short.MAX_VALUE);
        }
        track = new AudioTrack.Builder()
                .setAudioAttributes(new AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION).build())
                .setAudioFormat(new AudioFormat.Builder().setSampleRate(RATE).setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                        .setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
                .setTransferMode(AudioTrack.MODE_STATIC).setBufferSizeInBytes(n * 2).build();
        track.write(pcm, 0, n);
        log = new FileWriter(new File(dir, "beacon.csv"));
        log.write("t_ns,freq0_hz,freq1_hz,dur_ms\n");
        running = true;
        thread = new Thread(() -> {
            try {
                Thread.sleep(500);
                while (running) {
                    track.stop();
                    track.reloadStaticData();
                    long t = Clock.now();
                    track.play();
                    synchronized (this) { log.write(t + "," + (int) F0 + "," + (int) F1 + "," + (int) (DUR * 1000) + "\n"); }
                    Thread.sleep((long) (PERIOD * 1000));
                }
            } catch (Exception ignored) {
            }
        }, "beacon");
        thread.start();
    }

    void stop() {
        running = false;
        try { thread.join(2500); } catch (InterruptedException ignored) {}
        try { track.stop(); track.release(); } catch (Exception ignored) {}
        try { synchronized (this) { log.close(); } } catch (Exception ignored) {}
    }
}
