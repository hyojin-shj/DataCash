package com.datacash.wear;

import android.Manifest;
import android.app.Activity;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.Gravity;
import android.view.WindowManager;
import android.widget.LinearLayout;
import android.widget.TextView;

import org.json.JSONObject;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.Locale;

/**
 * 데이터캐시 워치: 켜 두면 서버(/api/live)를 보고 폰이 녹화를 시작하면 같은 세션 id 로 손목 기록을 시작하고,
 * 끝나면 파일을 서버에 올린다. 녹화 중 서버와 시각을 주고받는다 (watch_server_sync.csv) — 폰과 직접 연결할 필요가 없다.
 * 서버 주소·토큰은 빌드할 때 넣은 기본값 (local.properties). 바꿀 때만 adb 로: am start ... --es server URL --es token TOKEN (onCreate).
 */
public class WatchActivity extends Activity {
    private final Handler ui = new Handler(Looper.getMainLooper());
    private TextView title, state;
    private volatile String sid;
    private volatile boolean recording;
    private WristRecorder rec;
    private BufferedWriter syncLog;
    private long startedNs;

    private SharedPreferences prefs() { return getSharedPreferences("datacash", MODE_PRIVATE); }
    private String server() { return prefs().getString("server", BuildConfig.DEFAULT_SERVER); }
    private String token() { return prefs().getString("token", BuildConfig.DEFAULT_TOKEN); }

    @Override
    protected void onCreate(Bundle b) {
        super.onCreate(b);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        // 직접 켠 서버에 붙이기 (README 'Run locally'): adb shell am start -n com.datacash.wear/.WatchActivity --es server http://IP:8800 --es token TOKEN
        String srvExtra = getIntent().getStringExtra("server"), tokExtra = getIntent().getStringExtra("token");
        if (srvExtra != null && srvExtra.startsWith("http")) {
            SharedPreferences.Editor e = prefs().edit().putString("server", srvExtra.trim().replaceAll("/+$", ""));
            if (tokExtra != null) e.putString("token", tokExtra.trim());
            e.apply();
        }
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setGravity(Gravity.CENTER);
        GradientDrawable bg = new GradientDrawable(GradientDrawable.Orientation.TL_BR, new int[]{0xFFFF5F5F, 0xFFFFAF5F});
        root.setBackground(bg);
        title = new TextView(this);
        title.setText(Tr.s("데이터캐시", "DataCash"));
        title.setTextColor(Color.WHITE);
        title.setTextSize(16);
        title.setTypeface(Typeface.DEFAULT_BOLD);
        title.setGravity(Gravity.CENTER);
        state = new TextView(this);
        state.setTextColor(Color.WHITE);
        state.setTextSize(13);
        state.setGravity(Gravity.CENTER);
        state.setPadding(16, 8, 16, 0);
        root.addView(title);
        root.addView(state);
        setContentView(root);
        if (checkSelfPermission(Manifest.permission.BODY_SENSORS) != PackageManager.PERMISSION_GRANTED
                || checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.BODY_SENSORS, Manifest.permission.RECORD_AUDIO}, 1);
        }
        state.setText(Tr.s("폰에서 녹화를 시작하면\n손목 기록이 같이 시작돼요", "Start recording on\nyour phone"));
        new Thread(this::loop, "live").start();
    }

    private void loop() {
        while (!isFinishing()) {
            try {
                long a = WristRecorder.now();
                JSONObject live = get("/api/live?role=watch");
                long c = WristRecorder.now();
                String st = live == null ? "offline" : live.optString("state");
                if (recording) {
                    JSONObject t = get("/api/time");
                    long d = WristRecorder.now();
                    if (t != null && syncLog != null) synchronized (this) { syncLog.write(c + "," + t.getLong("ns") + "," + d + "\n"); }
                    long s = (WristRecorder.now() - startedNs) / 1_000_000_000L;
                    String hs = rec != null ? rec.healthState : "";
                    ui.post(() -> state.setText(String.format(Locale.US, Tr.s("● 손목 기록 %d:%02d", "● REC %d:%02d"), s / 60, s % 60) + (hs.isEmpty() ? "" : "\n" + hs)));
                    if (!"recording".equals(st) || live == null || !sid.equals(live.optString("sid"))) stopAndUpload();
                } else if ("recording".equals(st)) {
                    begin(live.getString("sid"));
                } else {
                    if (live == null) Discover.ensure(this);      // 맥 IP 가 바뀌었으면 서버를 다시 찾는다
                    ui.post(() -> state.setText(live == null ? Tr.s("서버에 연결 안 됨\n", "No server\n") + server() : Tr.s("폰에서 녹화를 시작하면\n손목 기록이 같이 시작돼요", "Start recording on\nyour phone")));
                }
                Thread.sleep(500);
            } catch (Exception e) {
                ui.post(() -> state.setText(Tr.s("오류: ", "Error: ") + e.getMessage()));
                try { Thread.sleep(1000); } catch (InterruptedException ignored) {}
            }
        }
    }

    private void begin(String id) throws Exception {
        sid = id;
        File dir = new File(getExternalFilesDir(null), "sessions/" + id);
        dir.mkdirs();
        syncLog = new BufferedWriter(new FileWriter(new File(dir, "watch_server_sync.csv")));
        syncLog.write("device_send_ns,server_ns,device_recv_ns\n");
        rec = new WristRecorder(this);
        rec.start(dir, prefs().getBoolean("bio", true));
        startedNs = WristRecorder.now();
        recording = true;
    }

    private void stopAndUpload() {
        recording = false;
        File dir = new File(getExternalFilesDir(null), "sessions/" + sid);
        try {
            JSONObject info = rec.stop();
            synchronized (this) { syncLog.close(); }
            info.put("model", Build.MANUFACTURER + " " + Build.MODEL).put("android", Build.VERSION.RELEASE);
            Files.write(new File(dir, "watch_meta.json"), info.toString(2));
            ui.post(() -> state.setText(Tr.s("올리는 중…", "Uploading…")));
            for (String n : new String[]{"watch_imu.csv", "watch_vib.csv", "watch_hr.csv", "watch_ibi.csv", "watch_skin.csv", "watch_audio.wav",
                    "watch_audio.json", "watch_server_sync.csv", "watch_meta.json"}) {
                File f = new File(dir, n);
                if (f.exists()) put("/api/sessions/" + sid + "/files/" + n, f);
            }
            post("/api/sessions/" + sid + "/watch_done");
            ui.post(() -> state.setText(Tr.s("올렸어요 ✓", "Uploaded ✓")));
        } catch (Exception e) {
            ui.post(() -> state.setText(Tr.s("올리기 실패: ", "Upload failed: ") + e.getMessage()));
        }
    }

    private HttpURLConnection open(String method, String path) throws Exception {
        HttpURLConnection h = (HttpURLConnection) new URL(server() + path).openConnection();
        h.setRequestMethod(method);
        h.setRequestProperty("X-DC-Token", token());
        h.setConnectTimeout(5000);           // 클라우드 서버(https)는 같은 와이파이보다 느리게 붙는다
        h.setReadTimeout(30000);
        return h;
    }

    private JSONObject get(String path) {
        try {
            HttpURLConnection h = open("GET", path);
            if (h.getResponseCode() != 200) return null;
            return new JSONObject(new String(h.getInputStream().readAllBytes(), "UTF-8"));
        } catch (Exception e) {
            return null;
        }
    }

    private void post(String path) throws Exception {
        HttpURLConnection h = open("POST", path);
        h.getResponseCode();
    }

    private void put(String path, File f) throws Exception {
        HttpURLConnection h = open("PUT", path);
        h.setDoOutput(true);
        h.setFixedLengthStreamingMode(f.length());
        try (OutputStream o = h.getOutputStream(); FileInputStream in = new FileInputStream(f)) {
            byte[] b = new byte[1 << 16];
            int k;
            while ((k = in.read(b)) > 0) o.write(b, 0, k);
        }
        if (h.getResponseCode() >= 400) throw new Exception(path + " " + h.getResponseCode());
    }
}
