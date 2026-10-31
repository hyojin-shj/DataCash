package com.datacash.wear;

import android.annotation.SuppressLint;
import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.AudioTimestamp;
import android.media.MediaRecorder;
import android.os.SystemClock;
import android.os.VibrationEffect;
import android.os.Vibrator;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.RandomAccessFile;
import java.util.Locale;

/**
 * 손목 기록 (워치 시계 BOOTTIME):
 *   watch_imu.csv   가속도(이벤트마다) + 최근 자이로 — 가장 빠른 속도
 *   watch_vib.csv   능동 진동 펄스: 0.5초마다 0.05초 울리고 명령 시각을 적는다 (울림 반응 → 쥐는 힘, 동기 신호도 겸함)
 *   watch_hr.csv    심박 t_ns,bpm,status — 생체 동의 녹화만. Samsung Health Sensor SDK 가 있으면 그것(연속 심박),
 *                   없거나 연결이 안 되면 안드로이드 기본 심박 센서 (status = -1)
 *   watch_ibi.csv   박동 간격 t_ns,ibi_ms,status — 삼성 SDK 가 있을 때만
 *   watch_skin.csv  피부온도 t_ns,temp_c,ambient_c,status — 삼성 SDK (또는 기기가 공개 센서로 줄 때)
 *   watch_audio.wav 마이크 (48kHz 가 되면 음향 비콘을 듣는다, 아니면 16kHz)
 */
final class WristRecorder implements SensorEventListener {
    static final long PULSE_MS = 50, PERIOD_MS = 500;
    private final Context ctx;
    private final SensorManager sm;
    private BufferedWriter imu, vib, hr, skin, ibi;
    private SamsungHealth samsung;
    private boolean stdHeart;
    volatile String healthState = "";
    private final float[] gyro = new float[3];
    private Thread pulses, audioThread;
    private volatile boolean running;
    private AudioRecord rec;
    private long audioStart = -1, audioFrames, nextAnchor;
    private final JSONArray anchors = new JSONArray();
    private int audioRate;
    private File dir;
    private final JSONObject info = new JSONObject();

    WristRecorder(Context c) {
        ctx = c;
        sm = (SensorManager) c.getSystemService(Context.SENSOR_SERVICE);
    }

    static long now() { return SystemClock.elapsedRealtimeNanos(); }

    void start(File d, boolean bio) throws Exception {
        dir = d;
        imu = writer("watch_imu.csv", "t_ns,ax,ay,az,gx,gy,gz");
        vib = writer("watch_vib.csv", "t_ns,dur_ms,amp");
        Sensor a = sm.getDefaultSensor(Sensor.TYPE_ACCELEROMETER), g = sm.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        if (g != null) sm.registerListener(this, g, SensorManager.SENSOR_DELAY_FASTEST);
        if (a != null) sm.registerListener(this, a, SensorManager.SENSOR_DELAY_FASTEST);
        info.put("accel", a != null ? a.getName() : JSONObject.NULL).put("gyro", g != null ? g.getName() : JSONObject.NULL);
        if (bio) {
            hr = writer("watch_hr.csv", "t_ns,bpm,status");
            if (SamsungHealth.available()) {           // 삼성 SDK: 연속 심박 + 박동 간격 + 피부온도
                ibi = writer("watch_ibi.csv", "t_ns,ibi_ms,status");
                skin = writer("watch_skin.csv", "t_ns,temp_c,ambient_c,status");
                samsung = new SamsungHealth(ctx, new SamsungHealth.Sink() {
                    @Override public void heartRate(long t, int bpm, int st) {
                        if (bpm > 0 && st == 1) write(hr, t + "," + bpm + "," + st + "\n");     // 1 = 측정 성공
                    }
                    @Override public void ibi(long t, int ms, int st) {
                        if (ms > 0 && st == 0) write(ibi, t + "," + ms + "," + st + "\n");       // 0 = 정상 박동
                    }
                    @Override public void skin(long t, float c, float amb, int st) {
                        boolean ok = st == 0 || (st == SamsungHealth.NO_STATUS && c > 20f && c < 45f);   // 0 = 측정 성공 · 상태가 없는 SDK 는 사람 피부 온도 범위만
                        if (ok) write(skin, t + "," + c + "," + amb + "," + st + "\n");
                    }
                    @Override public void state(String w) {
                        healthState = w;
                        if (w.contains("못") || w.contains("error") || w.contains("failed")) standardHeart();   // 연결이 안 되면 기본 심박 센서로
                    }
                });
                samsung.start(true, true);
            } else {
                standardHeart();
                Sensor sk = null;
                JSONArray names = new JSONArray();
                for (Sensor s : sm.getSensorList(Sensor.TYPE_ALL)) {
                    names.put(s.getStringType());
                    String t = s.getStringType().toLowerCase(Locale.US);
                    if (t.contains("skin") && t.contains("temp")) sk = s;
                }
                info.put("sensors", names).put("skin_temp", sk != null ? sk.getStringType() : JSONObject.NULL);
                if (sk != null) {
                    skin = writer("watch_skin.csv", "t_ns,temp_c,ambient_c,status");
                    sm.registerListener(this, sk, SensorManager.SENSOR_DELAY_NORMAL);
                }
                healthState = Tr.s("기본 심박 센서", "Basic HR sensor") + (sk != null ? Tr.s(" + 피부온도", " + skin temp") : Tr.s(" (피부온도는 삼성 SDK 필요)", " (skin temp needs Samsung SDK)"));
            }
        }
        running = true;
        Vibrator vb = ctx.getSystemService(Vibrator.class);
        pulses = new Thread(() -> {
            try {
                Thread.sleep(250);
                while (running) {
                    long t = now();
                    vb.vibrate(VibrationEffect.createOneShot(PULSE_MS, 255));
                    synchronized (this) { vib.write(t + "," + PULSE_MS + ",1.0\n"); }
                    Thread.sleep(PERIOD_MS);
                }
            } catch (Exception ignored) {
            }
        }, "pulses");
        pulses.start();
        startAudio();
    }

    @SuppressLint("MissingPermission")
    private void startAudio() {
        for (int rate : new int[]{48000, 16000}) {
            int min = AudioRecord.getMinBufferSize(rate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
            if (min <= 0) continue;
            try {
                AudioRecord r = new AudioRecord(MediaRecorder.AudioSource.MIC, rate, AudioFormat.CHANNEL_IN_MONO,
                        AudioFormat.ENCODING_PCM_16BIT, Math.max(min, rate / 5));
                if (r.getState() == AudioRecord.STATE_INITIALIZED) { rec = r; audioRate = rate; break; }
                r.release();
            } catch (Exception ignored) {
            }
        }
        if (rec == null) return;
        File out = new File(dir, "watch_audio.wav");
        rec.startRecording();
        audioThread = new Thread(() -> {
            byte[] buf = new byte[audioRate / 50 * 2];
            AudioTimestamp ts = new AudioTimestamp();
            try (FileOutputStream f = new FileOutputStream(out)) {
                f.write(new byte[44]);
                while (running) {
                    int n = rec.read(buf, 0, buf.length);
                    if (n > 0) { f.write(buf, 0, n); audioFrames += n / 2; }
                    // 첫 샘플 시각 + 1초마다 (샘플 번호, BOOTTIME) 기준점 — 마이크 샘플 시계의 어긋남·끊김을 서버가 바로잡는다
                    if ((audioStart < 0 || audioFrames >= nextAnchor) && rec.getTimestamp(ts, AudioTimestamp.TIMEBASE_BOOTTIME) == AudioRecord.SUCCESS) {
                        if (audioStart < 0) audioStart = ts.nanoTime - ts.framePosition * 1_000_000_000L / audioRate;
                        synchronized (anchors) { anchors.put(new JSONArray().put(ts.framePosition).put(ts.nanoTime)); }
                        nextAnchor = audioFrames + audioRate;
                    }
                }
            } catch (Exception ignored) {
            }
        }, "watch-audio");
        audioThread.start();
    }

    private synchronized void standardHeart() {
        if (stdHeart) return;
        Sensor h = sm.getDefaultSensor(Sensor.TYPE_HEART_RATE);
        if (h != null) sm.registerListener(this, h, SensorManager.SENSOR_DELAY_FASTEST);
        stdHeart = true;
        try { info.put("heart_rate", h != null ? "android TYPE_HEART_RATE" : JSONObject.NULL); } catch (Exception ignored) {}
    }

    private synchronized void write(BufferedWriter w, String line) {
        try { if (w != null && running) w.write(line); } catch (Exception ignored) {}
    }

    private BufferedWriter writer(String name, String header) throws Exception {
        BufferedWriter w = new BufferedWriter(new FileWriter(new File(dir, name)));
        w.write(header + "\n");
        return w;
    }

    @Override
    public synchronized void onSensorChanged(SensorEvent e) {
        try {
            int t = e.sensor.getType();
            if (t == Sensor.TYPE_GYROSCOPE) System.arraycopy(e.values, 0, gyro, 0, 3);
            else if (t == Sensor.TYPE_ACCELEROMETER)
                imu.write(String.format(Locale.US, "%d,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n", e.timestamp, e.values[0], e.values[1], e.values[2], gyro[0], gyro[1], gyro[2]));
            else if (t == Sensor.TYPE_HEART_RATE && hr != null && e.values[0] > 0)
                hr.write(e.timestamp + "," + e.values[0] + ",-1\n");
            else if (skin != null && samsung == null) skin.write(e.timestamp + "," + e.values[0] + ",NaN,-1\n");
        } catch (Exception ignored) {
        }
    }

    @Override public void onAccuracyChanged(Sensor s, int a) {}

    JSONObject stop() throws Exception {
        running = false;
        sm.unregisterListener(this);
        if (samsung != null) { samsung.stop(); info.put("samsung", samsung.info); }
        try { pulses.join(1500); } catch (InterruptedException ignored) {}
        if (rec != null) {
            try { audioThread.join(2000); } catch (InterruptedException ignored) {}
            rec.stop();
            rec.release();
            header(new File(dir, "watch_audio.wav"), audioFrames, audioRate);
            synchronized (anchors) {
                Files.write(new File(dir, "watch_audio.json"), new JSONObject().put("start_ns", audioStart > 0 ? audioStart : now())
                        .put("rate", audioRate).put("anchors", anchors).toString());
            }
        }
        synchronized (this) {
            for (BufferedWriter w : new BufferedWriter[]{imu, vib, hr, skin, ibi}) if (w != null) try { w.close(); } catch (Exception ignored) {}
        }
        info.put("health", healthState).put("audio_rate", audioRate).put("pulse_ms", PULSE_MS).put("period_ms", PERIOD_MS);
        return info;
    }

    private static void header(File f, long frames, int rate) throws Exception {
        long data = frames * 2;
        try (RandomAccessFile r = new RandomAccessFile(f, "rw")) {
            r.seek(0);
            r.writeBytes("RIFF"); r.write(le(36 + data)); r.writeBytes("WAVEfmt "); r.write(le(16));
            r.write(new byte[]{1, 0, 1, 0}); r.write(le(rate)); r.write(le(rate * 2L)); r.write(new byte[]{2, 0, 16, 0});
            r.writeBytes("data"); r.write(le(data));
        }
    }

    private static byte[] le(long v) { return new byte[]{(byte) v, (byte) (v >> 8), (byte) (v >> 16), (byte) (v >> 24)}; }
}
