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
    private TextView recBtn, markBtn, timer, hint;
    private final Handler ui = new Handler(Looper.getMainLooper());
    private String missionId, situation, title;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        missionId = getIntent().getStringExtra("mission");
        situation = getIntent().getStringExtra("situation");
        try { title = Store.mission(this, missionId).getString("title"); } catch (Exception e) { title = missionId; }
        FrameLayout root = new FrameLayout(this);
        preview = new TextureView(this);
        preview.setSurfaceTextureListener(this);
        root.addView(preview, new FrameLayout.LayoutParams(-1, -1));

        LinearLayout top = new LinearLayout(this);
        top.setOrientation(LinearLayout.VERTICAL);
        top.setPadding(Ui.dp(this, 18), Ui.dp(this, 36), Ui.dp(this, 18), Ui.dp(this, 12));
        top.setBackgroundColor(0x66000000);
        top.addView(Ui.text(this, title + " · " + situation, 17, Color.WHITE, true));
        timer = Ui.text(this, "준비", 14, Color.WHITE, false);
        top.addView(timer);
        root.addView(top, new FrameLayout.LayoutParams(-1, -2, Gravity.TOP));

        hint = Ui.text(this, "손과 물건이 잘 보이게 두고 녹화를 누르세요", 14, Color.WHITE, false);
        hint.setGravity(Gravity.CENTER);
        hint.setBackground(Ui.round(0x88000000, Ui.dp(this, 14)));
        hint.setPadding(Ui.dp(this, 14), Ui.dp(this, 8), Ui.dp(this, 14), Ui.dp(this, 8));
        FrameLayout.LayoutParams hp = new FrameLayout.LayoutParams(-2, -2, Gravity.CENTER_HORIZONTAL | Gravity.BOTTOM);
        hp.bottomMargin = Ui.dp(this, 200);
        root.addView(hint, hp);

        markBtn = Ui.text(this, "실수했어요", 16, Color.WHITE, true);
        markBtn.setGravity(Gravity.CENTER);
        markBtn.setBackground(Ui.round(0xAA000000, Ui.dp(this, 24)));
        markBtn.setPadding(Ui.dp(this, 22), Ui.dp(this, 12), Ui.dp(this, 22), Ui.dp(this, 12));
        markBtn.setVisibility(android.view.View.GONE);
        markBtn.setOnClickListener(v -> {
            if (!recording) return;
            marks.add(Clock.now());
            markBtn.setText(String.format(Locale.KOREA, "실수했어요 · %d", marks.size()));
        });
        FrameLayout.LayoutParams mp = new FrameLayout.LayoutParams(-2, -2, Gravity.CENTER_HORIZONTAL | Gravity.BOTTOM);
        mp.bottomMargin = Ui.dp(this, 140);
        root.addView(markBtn, mp);

        recBtn = Ui.text(this, "녹화", 18, Color.WHITE, true);
        recBtn.setGravity(Gravity.CENTER);
        recBtn.setBackground(Ui.gradient(Ui.dp(this, 40)));
        recBtn.setOnClickListener(v -> toggle());
        FrameLayout.LayoutParams rp = new FrameLayout.LayoutParams(Ui.dp(this, 160), Ui.dp(this, 64), Gravity.CENTER_HORIZONTAL | Gravity.BOTTOM);
        rp.bottomMargin = Ui.dp(this, 50);
        root.addView(recBtn, rp);
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
                    recBtn.setText("멈추기");
                    markBtn.setVisibility(android.view.View.VISIBLE);
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
        timer.setText(String.format(Locale.KOREA, "● 녹화 중 %d:%02d", s / 60, s % 60));
        ui.postDelayed(this::tick, 500);
    }

    private void stop() {
        busy = true;
        recording = false;
        markBtn.setVisibility(android.view.View.GONE);
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
                                .put("android", Build.VERSION.RELEASE)))
                        .put("participant", new JSONObject().put("consent", Store.consentJson(this)).put("age_band", Store.ageBand(this)))
                        .put("self_marks_ns", mk).put("video", video).put("audio", a).put("phone_imu_rows", imuRows)
                        .put("recorded_at", Clock.wallIso());
                Files.write(new File(dir, "meta.json"), meta.toString(2));
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
