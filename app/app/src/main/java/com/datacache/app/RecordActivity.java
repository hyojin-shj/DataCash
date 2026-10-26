package com.datacache.app;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.SurfaceTexture;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.Gravity;
import android.view.TextureView;
import android.view.WindowManager;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * 녹화: 카메라(장면 시각) + 마이크 48kHz + 폰 IMU + 음향 비콘, 녹화 중 '실수했어요'.
 * 끝나면 meta.json 을 쓰고 미션 후 질문으로. 워치(손목 IMU·진동·심박)는 워치 앱이 같은 세션 id 로 따로 기록한다.
 */
public class RecordActivity extends Activity implements TextureView.SurfaceTextureListener {
    private TextureView preview;
    private CameraRecorder camera;
    private AudioWav audio;
    private Imu imu;
    private Beacon beacon;
    private ServerSync sync;
    private File dir;
    private boolean recording, busy;
    private long startNs;
    private final List<Long> marks = new ArrayList<>();
    private TextView markBtn, timer, hint;
    private RecordRing recBtn;
    private String steps = "";
    private final Handler ui = new Handler(Looper.getMainLooper());
    private String missionId, situation, title;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        missionId = getIntent().getStringExtra("mission");
        situation = getIntent().getStringExtra("situation");
        try {
            JSONObject m = Store.mission(this, missionId);
            title = m.getString("title");
            JSONArray prims = m.optJSONArray("primitives");
            JSONObject ko = Store.primitivesKo(this);
            StringBuilder b = new StringBuilder();
            for (int i = 0; prims != null && i < prims.length(); i++) b.append(i > 0 ? " → " : "").append(ko.optString(prims.getString(i)));
            steps = b.toString();
        } catch (Exception e) {
            title = missionId;
        }
        FrameLayout root = new FrameLayout(this);
        preview = new TextureView(this);
        preview.setSurfaceTextureListener(this);
        root.addView(preview, new FrameLayout.LayoutParams(-1, -1));

        // 위: 반투명 둥근 안내 카드 (미션·보상 · 동작 순서 · 상황)
        LinearLayout top = new LinearLayout(this);
        top.setOrientation(LinearLayout.VERTICAL);
        top.setBackground(Ui.round(0x99000000, Ui.dp(this, 18)));
        top.setPadding(Ui.dp(this, 16), Ui.dp(this, 14), Ui.dp(this, 16), Ui.dp(this, 14));
        top.addView(Ui.text(this, String.format(Locale.KOREA, "%s 미션  +%,d원", title, Store.perMission(this)), 17, Color.WHITE, true));
        hint = Ui.text(this, steps.isEmpty() ? "손과 물건이 잘 보이게 두고 빨간 버튼을 누르세요" : "동작 순서 · " + steps, 13, Ui.ACCENT, true);
        hint.setPadding(0, Ui.dp(this, 8), 0, 0);
        top.addView(hint);
        TextView sit = Ui.text(this, "상황 · " + situation, 13, 0xDDFFFFFF, false);
        sit.setPadding(0, Ui.dp(this, 6), 0, 0);
        top.addView(sit);
        FrameLayout.LayoutParams tp = new FrameLayout.LayoutParams(-1, -2, Gravity.TOP);
        tp.setMargins(Ui.dp(this, 12), Ui.dp(this, 40), Ui.dp(this, 12), 0);
        root.addView(top, tp);

        // 아래: 검은 띠 — 왼쪽 '실수했어요', 가운데 녹화 버튼, 오른쪽 시간
        FrameLayout bar = new FrameLayout(this);
        bar.setBackgroundColor(0xE6000000);
        markBtn = Ui.text(this, "실수했어요", 13, Color.WHITE, true);
        markBtn.setGravity(Gravity.CENTER);
        markBtn.setBackground(Ui.pressable(0x33FFFFFF, Ui.dp(this, 22)));
        markBtn.setPadding(Ui.dp(this, 14), Ui.dp(this, 10), Ui.dp(this, 14), Ui.dp(this, 10));
        markBtn.setAlpha(0.4f);
        markBtn.setOnClickListener(v -> {
            if (!recording) return;
            marks.add(Clock.now());
            markBtn.setText(String.format(Locale.KOREA, "실수했어요 %d", marks.size()));
        });
        FrameLayout.LayoutParams mp = new FrameLayout.LayoutParams(-2, -2, Gravity.CENTER_VERTICAL | Gravity.START);
        mp.leftMargin = Ui.dp(this, 20);
        bar.addView(markBtn, mp);
        recBtn = new RecordRing(this);
        recBtn.setCameraStyle();
        recBtn.setContentDescription("녹화");
        recBtn.setOnClickListener(v -> toggle());
        bar.addView(recBtn, new FrameLayout.LayoutParams(Ui.dp(this, 76), Ui.dp(this, 76), Gravity.CENTER));
        timer = Ui.text(this, "0:00", 16, Color.WHITE, true);
        FrameLayout.LayoutParams tm = new FrameLayout.LayoutParams(-2, -2, Gravity.CENTER_VERTICAL | Gravity.END);
        tm.rightMargin = Ui.dp(this, 28);
        bar.addView(timer, tm);
        root.addView(bar, new FrameLayout.LayoutParams(-1, Ui.dp(this, 128), Gravity.BOTTOM));
        setContentView(root);

        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED
                || checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO}, 1);
        }
    }

    @Override
    public void onRequestPermissionsResult(int code, String[] perms, int[] res) {
        if (preview.isAvailable()) openCamera(preview.getSurfaceTexture());
    }

    @Override public void onSurfaceTextureAvailable(SurfaceTexture st, int w, int h) { openCamera(st); }
    @Override public void onSurfaceTextureSizeChanged(SurfaceTexture st, int w, int h) {}
    @Override public boolean onSurfaceTextureDestroyed(SurfaceTexture st) { return true; }
    @Override public void onSurfaceTextureUpdated(SurfaceTexture st) {}

    private void openCamera(SurfaceTexture st) {
        if (camera != null || checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) return;
        camera = new CameraRecorder(this);
        camera.open(st, new CameraRecorder.Ready() {
            @Override public void ok() {}
            @Override public void fail(String why) { ui.post(() -> Toast.makeText(RecordActivity.this, "카메라: " + why, Toast.LENGTH_LONG).show()); }
        });
    }

    private void toggle() {
        if (busy || camera == null) return;
        if (!recording) start(); else stop();
    }

    private void start() {
        busy = true;
        String sid = Clock.sessionId();
        dir = new File(Store.sessionsDir(this), sid);
        dir.mkdirs();
        try {
            audio = new AudioWav();
            audio.start(dir);
            imu = new Imu(this);
            imu.start(dir);
            beacon = new Beacon();
            beacon.start(dir);
            sync = new ServerSync(this);   // 서버에 녹화 시작을 알리고 시각을 주고받는다 (워치가 같이 기록)
            sync.start(dir);
        } catch (Exception e) {
            Toast.makeText(this, "시작 실패: " + e.getMessage(), Toast.LENGTH_LONG).show();
            busy = false;
            return;
        }
        camera.start(dir, new CameraRecorder.Ready() {
            @Override public void ok() {
                ui.post(() -> {
                    recording = true;
                    busy = false;
                    startNs = Clock.now();
                    marks.clear();
                    recBtn.setRecording(true);
                    markBtn.setAlpha(1f);
                    hint.setText("실수하거나 다시 하면 '실수했어요'를 눌러 주세요");
                    tick();
                });
            }
            @Override public void fail(String why) {
                ui.post(() -> { busy = false; Toast.makeText(RecordActivity.this, "녹화 실패: " + why, Toast.LENGTH_LONG).show(); });
            }
        });
    }

    private void tick() {
        if (!recording) return;
        long s = (Clock.now() - startNs) / 1_000_000_000L;
        timer.setText(String.format(Locale.KOREA, "● %d:%02d", s / 60, s % 60));
        ui.postDelayed(this::tick, 500);
    }

    private void stop() {
        busy = true;
        recording = false;
        recBtn.setRecording(false);
        markBtn.setAlpha(0.4f);
        new Thread(() -> {
            try {
                JSONObject video = camera.stop(dir);
                beacon.stop();
                sync.stop();
                JSONObject a = audio.stop(dir);
                long imuRows = imu.stop();
                JSONArray mk = new JSONArray();
                for (long t : marks) mk.put(t);
                JSONObject meta = new JSONObject()
                        .put("session_id", dir.getName()).put("format", "datacache/1").put("clock", "BOOTTIME")
                        .put("mission", new JSONObject().put("id", missionId).put("title", title).put("situation", situation))
                        .put("devices", new JSONObject().put("phone", new JSONObject().put("model", Build.MANUFACTURER + " " + Build.MODEL)
                                .put("android", Build.VERSION.RELEASE).put("id", Store.deviceId(this))))
                        .put("participant", new JSONObject().put("consent", Store.consentJson(this)).put("age_band", Store.ageBand(this)))
                        .put("self_marks_ns", mk).put("video", video).put("audio", a).put("phone_imu_rows", imuRows)
                        .put("recorded_at", Clock.wallIso());
                Files.write(new File(dir, "meta.json"), meta.toString(2));
                Store.markSituation(this, missionId, situation);
                ui.post(() -> {
                    startActivity(new Intent(this, SurveyActivity.class).putExtra("dir", dir.getAbsolutePath()));
                    finish();
                });
            } catch (Exception e) {
                ui.post(() -> { busy = false; Toast.makeText(this, "저장 실패: " + e.getMessage(), Toast.LENGTH_LONG).show(); });
            }
        }).start();
    }

    @Override
    protected void onDestroy() {
        if (recording) { try { camera.stop(dir); beacon.stop(); audio.stop(dir); imu.stop(); } catch (Exception ignored) {} }
        if (camera != null) camera.close();
        super.onDestroy();
    }
}
