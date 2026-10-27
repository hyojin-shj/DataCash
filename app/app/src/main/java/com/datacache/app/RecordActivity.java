package com.datacache.app;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PointF;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;
import android.graphics.RectF;
import android.graphics.SurfaceTexture;
import android.graphics.Typeface;
import android.media.FaceDetector;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.Gravity;
import android.view.ScaleGestureDetector;
import android.view.TextureView;
import android.view.View;
import android.view.WindowManager;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * 촬영 화면: 카메라(장면 시각) + 마이크 48kHz + 폰 IMU + 음향 비콘 + 서버 시각 (워치는 서버를 보고 같이 기록).
 *
 * 화면: 미리보기가 화면 전체(비율 유지), 위에 미션·상태 카드, 일반 동영상 앱처럼 가운데 둥근 녹화 버튼과
 * 왼쪽 타이머(끔·3·5·10초, 누를 때마다 바뀜 — 고른 만큼 센 뒤 시작), 오른쪽 앞/뒤 카메라 전환.
 * 세로면 아래 가운데, 가로면 오른쪽 가운데. 두 손가락으로 확대·축소, 배율 버튼으로 빠르게 (녹화 중 바뀌면 시각을 남김).
 *
 * 녹화는 두 단계: 1단계 '얼굴 확인' — 정면 얼굴이 동그라미 안에 1초 이어서 보이면 통과 (폰 안의 얼굴 찾기 +
 * 서버에 등록한 본인과 맞춰 봄), 10초가 지나면 '얼굴 없이 진행'. 2단계 '동작 촬영' — '실수했어요' 버튼.
 * 서버는 1단계 얼굴로 참여자를 알아보고, 1단계 구간은 판매 데이터에서 뺀다 (meta.json face_check).
 */
public class RecordActivity extends Activity implements TextureView.SurfaceTextureListener {
    private static final int[] TIMERS = {0, 3, 5, 10};
    private static final int FACE_STREAK = 5;              // 0.2초 × 5 = 1초 이어서
    private static final long FACE_SKIP_MS = 10_000, FACE_MISMATCH_MS = 4_000;

    private final Handler ui = new Handler(Looper.getMainLooper());
    private final ExecutorService faceWorker = Executors.newSingleThreadExecutor();
    private SharedPreferences prefs;
    private TextureView preview;
    private CameraRecorder camera;
    private AudioWav audio;
    private Imu imu;
    private Beacon beacon;
    private ServerSync sync;
    private File dir;
    private boolean recording, busy, front;
    private long startNs;
    private final List<Long> marks = new ArrayList<>();
    private String missionId, situation, title, steps = "";

    private LinearLayout top, controls, zoomBar;
    private TextView status, countdown, timerBtn, markBtn;
    private RecordRing recBtn;
    private View switchBtn;
    private int timerSeconds, countdownLeft;

    // 1단계 얼굴 확인
    private FrameLayout faceLayer;
    private FaceOval faceOval;
    private TextView faceTitle, faceSub, faceSkip;
    private boolean faceChecking, facePassed;
    private int faceStreak;
    private long faceStartMs, faceEndNs, lastIdMs;
    private String identity = "pending";   // pending · checked · mismatch · no_registry · offline
    private double identityScore = -1;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        prefs = getSharedPreferences("datacache", MODE_PRIVATE);
        timerSeconds = prefs.getInt("rec_timer", 0);
        front = prefs.getBoolean("rec_front", false);
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
        build();
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED
                || checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO}, 1);
        }
    }

    // ---------- 화면 ----------

    private void build() {
        FrameLayout root = new FrameLayout(this);
        root.setBackgroundColor(Color.BLACK);
        preview = new TextureView(this);
        preview.setSurfaceTextureListener(this);
        Ui.fitPreview(this, preview);
        root.addView(preview, new FrameLayout.LayoutParams(-1, -1));
        ScaleGestureDetector pinch = new ScaleGestureDetector(this, new ScaleGestureDetector.SimpleOnScaleGestureListener() {
            @Override public boolean onScale(ScaleGestureDetector d) {
                if (camera != null) { camera.setZoom(camera.zoom() * d.getScaleFactor()); paintZoom(); }
                return true;
            }
        });
        preview.setOnTouchListener((v, e) -> { pinch.onTouchEvent(e); return true; });

        faceLayer = new FrameLayout(this);
        faceLayer.setVisibility(View.GONE);
        faceOval = new FaceOval(this);
        faceLayer.addView(faceOval, new FrameLayout.LayoutParams(-1, -1));
        LinearLayout box = new LinearLayout(this);
        box.setOrientation(LinearLayout.VERTICAL);
        box.setGravity(Gravity.CENTER_HORIZONTAL);
        faceTitle = shadowText(22, true);
        box.addView(faceTitle);
        faceSub = shadowText(15, false);
        faceSub.setPadding(0, Ui.dp(this, 6), 0, 0);
        box.addView(faceSub);
        faceSkip = Ui.text(this, "얼굴 없이 진행", 14, Color.WHITE, true);
        faceSkip.setPadding(Ui.dp(this, 18), Ui.dp(this, 10), Ui.dp(this, 18), Ui.dp(this, 10));
        faceSkip.setBackground(Ui.round(0x88000000, Ui.dp(this, 20)));
        faceSkip.setVisibility(View.GONE);
        faceSkip.setOnClickListener(v -> endFaceCheck(false));
        box.addView(faceSkip, Ui.margins(this, -2, -2, 0, 0, 14, 0, 0));
        FrameLayout.LayoutParams bp = new FrameLayout.LayoutParams(-1, -2, Gravity.BOTTOM);
        bp.bottomMargin = Ui.dp(this, 150);
        faceLayer.addView(box, bp);
        root.addView(faceLayer, new FrameLayout.LayoutParams(-1, -1));

        top = new LinearLayout(this);
        top.setOrientation(LinearLayout.VERTICAL);
        top.setBackground(Ui.round(0x99000000, Ui.dp(this, 18)));
        top.setPadding(Ui.dp(this, 16), Ui.dp(this, 14), Ui.dp(this, 16), Ui.dp(this, 14));
        top.addView(Ui.text(this, String.format(Locale.KOREA, "%s 미션  +%,d원", title, Store.perMission(this)), 17, Color.WHITE, true));
        TextView seq = Ui.text(this, steps.isEmpty() ? "1단계 얼굴 확인 → 2단계 동작 촬영" : "1단계 얼굴 확인 → 2단계 동작 촬영 · " + steps, 13, Ui.ACCENT, true);
        seq.setPadding(0, Ui.dp(this, 8), 0, 0);
        top.addView(seq);
        status = Ui.text(this, "상황 · " + situation, 13, 0xDDFFFFFF, false);
        status.setPadding(0, Ui.dp(this, 6), 0, 0);
        top.addView(status);
        root.addView(top);

        countdown = shadowText(96, true);
        countdown.setGravity(Gravity.CENTER);
        countdown.setVisibility(View.GONE);
        root.addView(countdown, new FrameLayout.LayoutParams(-1, -1, Gravity.CENTER));

        zoomBar = new LinearLayout(this);
        zoomBar.setGravity(Gravity.CENTER);
        zoomBar.setPadding(Ui.dp(this, 6), Ui.dp(this, 6), Ui.dp(this, 6), Ui.dp(this, 6));
        zoomBar.setBackground(Ui.round(0x55000000, Ui.dp(this, 24)));
        zoomBar.setVisibility(View.GONE);
        root.addView(zoomBar);

        controls = new LinearLayout(this);
        controls.setGravity(Gravity.CENTER);
        controls.setPadding(Ui.dp(this, 16), Ui.dp(this, 16), Ui.dp(this, 16), Ui.dp(this, 24));
        timerBtn = Ui.text(this, "", 14, Color.WHITE, true);
        timerBtn.setGravity(Gravity.CENTER);
        timerBtn.setBackground(Ui.pressable(0x55000000, Ui.dp(this, 26)));
        timerBtn.setOnClickListener(v -> cycleTimer());
        paintTimer();
        recBtn = new RecordRing(this);
        recBtn.setCameraStyle();
        recBtn.setContentDescription("녹화");
        recBtn.setOnClickListener(v -> press());
        FrameLayout sw = new FrameLayout(this);
        sw.setBackground(Ui.pressable(0x55000000, Ui.dp(this, 26)));
        Icon swi = new Icon(this, Icon.SWITCH, Color.WHITE);
        sw.addView(swi, new FrameLayout.LayoutParams(Ui.dp(this, 26), Ui.dp(this, 26), Gravity.CENTER));
        sw.setContentDescription("앞뒤 카메라 바꾸기");
        sw.setOnClickListener(v -> switchCamera());
        switchBtn = sw;
        controls.addView(timerBtn, new LinearLayout.LayoutParams(Ui.dp(this, 52), Ui.dp(this, 52)));
        controls.addView(recBtn, new LinearLayout.LayoutParams(Ui.dp(this, 78), Ui.dp(this, 78)));
        controls.addView(switchBtn, new LinearLayout.LayoutParams(Ui.dp(this, 52), Ui.dp(this, 52)));
        root.addView(controls);

        markBtn = Ui.text(this, "실수했어요", 15, Color.WHITE, true);
        markBtn.setPadding(Ui.dp(this, 18), Ui.dp(this, 10), Ui.dp(this, 18), Ui.dp(this, 10));
        markBtn.setBackground(Ui.pressable(0x99000000, Ui.dp(this, 22)));
        markBtn.setContentDescription("방금 실수했거나 다시 하는 중이면 눌러 주세요");
        markBtn.setVisibility(View.GONE);
        markBtn.setOnClickListener(v -> {
            if (!recording) return;
            marks.add(Clock.now());
            markBtn.setText(String.format(Locale.KOREA, "실수했어요 · %d", marks.size()));
        });
        FrameLayout.LayoutParams mp = new FrameLayout.LayoutParams(-2, -2, Gravity.BOTTOM | Gravity.CENTER_HORIZONTAL);
        mp.bottomMargin = Ui.dp(this, 130);
        root.addView(markBtn, mp);
        setContentView(root);
        applyLayout();
    }

    private TextView shadowText(float sp, boolean bold) {
        TextView t = Ui.text(this, "", sp, Color.WHITE, bold);
        t.setGravity(Gravity.CENTER);
        t.setShadowLayer(10, 0, 0, Color.BLACK);
        return t;
    }

    /** 세로: 버튼 줄 아래 가운데 · 가로: 오른쪽 가운데 세로 줄. */
    private void applyLayout() {
        boolean land = getResources().getConfiguration().orientation == Configuration.ORIENTATION_LANDSCAPE;
        controls.setOrientation(land ? LinearLayout.VERTICAL : LinearLayout.HORIZONTAL);
        int gap = Ui.dp(this, 34);
        for (int i = 0; i < controls.getChildCount(); i++) {
            LinearLayout.LayoutParams p = (LinearLayout.LayoutParams) controls.getChildAt(i).getLayoutParams();
            p.setMargins(land ? 0 : gap / 2, land ? gap / 2 : 0, land ? 0 : gap / 2, land ? gap / 2 : 0);
        }
        controls.setLayoutParams(new FrameLayout.LayoutParams(land ? -2 : -1, land ? -1 : -2,
                land ? Gravity.END | Gravity.CENTER_VERTICAL : Gravity.BOTTOM | Gravity.CENTER_HORIZONTAL));
        zoomBar.setOrientation(land ? LinearLayout.VERTICAL : LinearLayout.HORIZONTAL);
        FrameLayout.LayoutParams zp = new FrameLayout.LayoutParams(-2, -2,
                land ? Gravity.END | Gravity.CENTER_VERTICAL : Gravity.BOTTOM | Gravity.CENTER_HORIZONTAL);
        if (land) zp.rightMargin = Ui.dp(this, 120); else zp.bottomMargin = Ui.dp(this, 128);
        zoomBar.setLayoutParams(zp);
        FrameLayout.LayoutParams tp = new FrameLayout.LayoutParams(-1, -2, Gravity.TOP);
        tp.setMargins(Ui.dp(this, 12), Ui.dp(this, 36), land ? Ui.dp(this, 120) : Ui.dp(this, 12), 0);
        top.setLayoutParams(tp);
        FrameLayout.LayoutParams mp = (FrameLayout.LayoutParams) markBtn.getLayoutParams();
        mp.gravity = land ? Gravity.BOTTOM | Gravity.START : Gravity.BOTTOM | Gravity.CENTER_HORIZONTAL;
        mp.bottomMargin = land ? Ui.dp(this, 20) : Ui.dp(this, 130);
        mp.leftMargin = land ? Ui.dp(this, 20) : 0;
        markBtn.setLayoutParams(mp);
    }

    @Override
    public void onConfigurationChanged(Configuration c) {
        super.onConfigurationChanged(c);
        applyLayout();
        Ui.applyFit(this, preview);
    }

    // ---------- 카메라 ----------

    @Override
    public void onRequestPermissionsResult(int code, String[] perms, int[] res) {
        if (preview.isAvailable()) openCamera(preview.getSurfaceTexture());
    }

    @Override public void onSurfaceTextureAvailable(SurfaceTexture st, int w, int h) { openCamera(st); }
    @Override public void onSurfaceTextureSizeChanged(SurfaceTexture st, int w, int h) { Ui.applyFit(this, preview); }
    @Override public boolean onSurfaceTextureDestroyed(SurfaceTexture st) { return true; }
    @Override public void onSurfaceTextureUpdated(SurfaceTexture st) {}

    private void openCamera(SurfaceTexture st) {
        if (camera != null || checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) return;
        camera = new CameraRecorder(this);
        camera.open(st, new CameraRecorder.Ready() {
            @Override public void ok() { ui.post(() -> { buildZoomBar(); Ui.applyFit(RecordActivity.this, preview); }); }
            @Override public void fail(String why) { ui.post(() -> Toast.makeText(RecordActivity.this, "카메라: " + why, Toast.LENGTH_LONG).show()); }
        }, front);
    }

    private void switchCamera() {
        if (recording || busy || countdownLeft > 0) return;
        front = !front;
        prefs.edit().putBoolean("rec_front", front).apply();
        if (camera != null) camera.close();
        camera = null;
        if (preview.isAvailable()) openCamera(preview.getSurfaceTexture());
        say(front ? "앞 카메라" : "뒤 카메라");
    }

    /** 배율 버튼: 이 카메라가 되는 것만 (0.5 · 1 · 2 · 5). */
    private void buildZoomBar() {
        zoomBar.removeAllViews();
        if (camera == null) return;
        for (float z : new float[]{0.5f, 1f, 2f, 5f}) {
            if (z < camera.minZoom() - 0.01f || z > camera.maxZoom() + 0.01f) continue;
            TextView b = Ui.text(this, (z < 1 ? ".5" : String.valueOf((int) z)) + "×", 13, Color.WHITE, true);
            b.setGravity(Gravity.CENTER);
            b.setTag(z);
            b.setOnClickListener(v -> { camera.setZoom((Float) v.getTag()); paintZoom(); });
            zoomBar.addView(b, Ui.margins(this, Ui.dp(this, 40), Ui.dp(this, 40), 0, 3, 3, 3, 3));
        }
        zoomBar.setVisibility(zoomBar.getChildCount() > 1 && !faceChecking ? View.VISIBLE : View.GONE);
        paintZoom();
    }

    private void paintZoom() {
        if (camera == null) return;
        float z = camera.zoom();
        for (int i = 0; i < zoomBar.getChildCount(); i++) {
            TextView b = (TextView) zoomBar.getChildAt(i);
            boolean on = Math.abs((Float) b.getTag() - z) < 0.05f;
            b.setTextColor(on ? Ui.ACCENT : Color.WHITE);
            b.setBackground(Ui.round(on ? 0xAA000000 : 0x33000000, Ui.dp(this, 20)));
            if (on || i == 0) b.setText(on && Math.abs(z - Math.round(z)) > 0.05f && z > 1
                    ? String.format(Locale.US, "%.1f×", z) : b.getText());
        }
    }

    // ---------- 타이머 ----------

    private void cycleTimer() {
        if (recording || countdownLeft > 0) return;
        int i = 0;
        while (i < TIMERS.length && TIMERS[i] != timerSeconds) i++;
        timerSeconds = TIMERS[(i + 1) % TIMERS.length];
        prefs.edit().putInt("rec_timer", timerSeconds).apply();
        paintTimer();
        say(timerSeconds == 0 ? "타이머 끔" : "녹화 버튼을 누르면 " + timerSeconds + "초 뒤에 시작해요");
    }

    private void paintTimer() {
        timerBtn.setText(timerSeconds == 0 ? "타이머\n끔" : timerSeconds + "초");
        timerBtn.setTextSize(timerSeconds == 0 ? 10 : 15);
        timerBtn.setContentDescription(timerSeconds == 0 ? "타이머 끔" : "타이머 " + timerSeconds + "초");
    }

    private void press() {
        if (busy || camera == null) return;
        if (recording) { stop(); return; }
        if (countdownLeft > 0) { cancelCountdown(); return; }
        if (timerSeconds == 0) { start(); return; }
        countdownLeft = timerSeconds;
        countdown.setVisibility(View.VISIBLE);
        say("다시 누르면 취소돼요");
        countdownTick();
    }

    private final Runnable tickRunner = this::countdownTick;

    private void countdownTick() {
        if (countdownLeft <= 0) return;
        countdown.setText(String.valueOf(countdownLeft));
        countdownLeft--;
        if (countdownLeft == 0) {
            ui.postDelayed(() -> { countdown.setVisibility(View.GONE); start(); }, 1000);
        } else {
            ui.postDelayed(tickRunner, 1000);
        }
    }

    private void cancelCountdown() {
        countdownLeft = 0;
        ui.removeCallbacks(tickRunner);
        countdown.setVisibility(View.GONE);
        say("타이머를 취소했어요");
    }

    private void say(String s) {
        Toast.makeText(this, s, Toast.LENGTH_SHORT).show();
    }

    // ---------- 녹화 ----------

    @SuppressWarnings("deprecation")
    private void start() {
        android.media.AudioManager am = (android.media.AudioManager) getSystemService(AUDIO_SERVICE);
        if (am != null && (am.getMode() == android.media.AudioManager.MODE_IN_CALL
                || am.getMode() == android.media.AudioManager.MODE_IN_COMMUNICATION)) {
            Toast.makeText(this, "통화 중에는 소리가 녹음되지 않아요. 통화를 끝내고 다시 눌러 주세요", Toast.LENGTH_LONG).show();
            return;
        }
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
        int rot = getWindowManager().getDefaultDisplay().getRotation();
        camera.start(dir, rot, new CameraRecorder.Ready() {
            @Override public void ok() {
                ui.post(() -> {
                    recording = true;
                    busy = false;
                    startNs = Clock.now();
                    marks.clear();
                    recBtn.setRecording(true);
                    timerBtn.setAlpha(0.3f);
                    switchBtn.setAlpha(0.3f);
                    tick();
                    startFaceCheck();
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
        boolean quiet = s >= 4 && audio != null && audio.peak < 30;     // 마이크가 막혔거나 다른 앱이 쓰는 중
        status.setText(String.format(Locale.KOREA, "● 녹화 %d:%02d · 상황 · %s", s / 60, s % 60, situation)
                + (quiet ? "\n⚠ 소리가 거의 안 들려요 — 마이크를 가리지 않았는지 봐 주세요" : ""));
        ui.postDelayed(this::tick, 500);
    }

    // ---------- 1단계 얼굴 확인 ----------

    private void startFaceCheck() {
        faceChecking = true;
        facePassed = false;
        faceStreak = 0;
        faceEndNs = 0;
        identity = "pending";
        identityScore = -1;
        lastIdMs = 0;
        faceStartMs = System.currentTimeMillis();
        faceOval.setOk(false);
        faceOval.setVisibility(View.VISIBLE);
        faceSkip.setVisibility(View.GONE);
        faceTitle.setText("1단계 · 얼굴 확인");
        faceSub.setText("카메라 가까이에서 정면을 봐 주세요\n얼굴이 동그라미 안에 오게");
        faceLayer.setVisibility(View.VISIBLE);
        zoomBar.setVisibility(View.GONE);
        markBtn.setVisibility(View.GONE);
        ui.post(this::faceLoop);
    }

    /** 미리보기를 작게 떠서(0.2초마다) 폰 안에서 정면 얼굴을 찾고, 찾으면 1초에 한 번 서버에 본인인지 묻는다. */
    private void faceLoop() {
        if (!faceChecking || !recording) return;
        if (System.currentTimeMillis() - faceStartMs > FACE_SKIP_MS) faceSkip.setVisibility(View.VISIBLE);
        int w = 480, h = preview.getWidth() > 0 ? Math.round(480f * preview.getHeight() / preview.getWidth()) : 640;
        Bitmap bm = preview.getBitmap(w, h & ~1);
        if (bm == null) { ui.postDelayed(this::faceLoop, 200); return; }
        RectF oval = faceOval.ovalIn(w, h);
        faceWorker.execute(() -> {
            boolean found = frontalIn(bm, oval);
            if (found && System.currentTimeMillis() - lastIdMs > 1000 && !identity.equals("checked")) {
                lastIdMs = System.currentTimeMillis();
                askServer(bm);
            }
            bm.recycle();
            ui.postDelayed(() -> onFace(found), 200);
        });
    }

    /** Android 기본 얼굴 찾기(두 눈 기준 — 정면에 가까운 얼굴만 찾힌다): 동그라미 안, 너무 작지 않게. */
    static boolean frontalIn(Bitmap src, RectF oval) {
        Bitmap b = src.copy(Bitmap.Config.RGB_565, false);
        try {
            FaceDetector.Face[] faces = new FaceDetector.Face[3];
            int n = new FaceDetector(b.getWidth(), b.getHeight(), faces.length).findFaces(b, faces);
            PointF mid = new PointF();
            for (int i = 0; i < n; i++) {
                FaceDetector.Face f = faces[i];
                f.getMidPoint(mid);
                if (f.confidence() >= 0.4f && f.eyesDistance() >= oval.width() * 0.18f && oval.contains(mid.x, mid.y)) return true;
            }
            return false;
        } finally {
            b.recycle();
        }
    }

    private void askServer(Bitmap bm) {
        ByteArrayOutputStream jpg = new ByteArrayOutputStream();
        bm.compress(Bitmap.CompressFormat.JPEG, 88, jpg);
        byte[] data = jpg.toByteArray();
        new Thread(() -> {
            try {
                JSONObject r = Uploader.call(this, "POST", "/api/face/check", data, "image/jpeg");
                boolean reg = r.optBoolean("registered");
                Store.setOwnerFace(this, reg);
                if (!reg) identity = "no_registry";
                else {
                    identityScore = Math.max(identityScore, r.optDouble("similarity", -1));
                    identity = r.optBoolean("match") ? "checked" : "mismatch";
                }
            } catch (Exception e) {
                if (identity.equals("pending")) identity = Uploader.reason(e).contains("얼굴") ? "pending" : "offline";
            }
        }).start();
    }

    private void onFace(boolean found) {
        if (!faceChecking) return;
        boolean idOk = identity.equals("checked") || identity.equals("no_registry") || identity.equals("offline");
        faceStreak = found ? faceStreak + 1 : 0;
        faceOval.setOk(found && idOk);
        long waited = System.currentTimeMillis() - faceStartMs;
        if (found && idOk) faceSub.setText("좋아요, 그대로 잠깐만요");
        else if (found && identity.equals("mismatch") && waited > FACE_MISMATCH_MS)
            faceSub.setText("등록한 얼굴과 달라 보여요\n밝은 곳에서 안경·모자를 벗고 정면으로");
        else if (found) faceSub.setText("본인인지 확인하는 중…");
        else faceSub.setText("카메라 가까이에서 정면을 봐 주세요\n얼굴이 동그라미 안에 오게");
        if (faceStreak >= FACE_STREAK && idOk) endFaceCheck(true);
        else faceLoop();
    }

    /** 1단계 끝 (통과 또는 건너뜀) → 2단계 안내를 잠깐 보여 주고 사라진다. */
    private void endFaceCheck(boolean passed) {
        if (!faceChecking) return;
        faceChecking = false;
        facePassed = passed;
        faceEndNs = Clock.now();
        faceSkip.setVisibility(View.GONE);
        faceOval.setVisibility(passed ? View.VISIBLE : View.GONE);
        faceTitle.setText(!passed ? "얼굴 없이 진행해요" : identity.equals("checked") ? "✓ 본인 확인됐어요" : "✓ 얼굴 확인됐어요");
        faceSub.setText(passed ? "2단계 · 이제 동작을 시작하세요"
                : "등록된 본인으로 알아보지 못하면 영상에서 가려질 수 있어요\n2단계 · 동작을 시작하세요");
        markBtn.setVisibility(View.VISIBLE);
        ui.postDelayed(() -> faceOval.setVisibility(View.GONE), 700);
        ui.postDelayed(() -> {
            if (!faceChecking) {
                faceLayer.setVisibility(View.GONE);
                if (zoomBar.getChildCount() > 1) zoomBar.setVisibility(View.VISIBLE);
            }
        }, 2500);
    }

    private JSONObject faceCheckInfo() throws Exception {
        JSONObject f = new JSONObject().put("passed", facePassed).put("start_ns", startNs).put("clock", "BOOTTIME")
                .put("detector", "android.media.FaceDetector").put("identity", identity);
        if (faceEndNs > 0) f.put("end_ns", faceEndNs);
        if (identityScore >= 0) f.put("identity_score", Math.round(identityScore * 1000) / 1000.0);
        return f;
    }

    // ---------- 멈추기 ----------

    private void stop() {
        busy = true;
        recording = false;
        if (faceChecking) { faceChecking = false; faceEndNs = Clock.now(); }
        recBtn.setRecording(false);
        markBtn.setVisibility(View.GONE);
        faceLayer.setVisibility(View.GONE);
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
                        .put("participant", new JSONObject().put("consent", Store.consentJson(this)).put("age_band", Store.ageBand(this))
                                .put("gender", Store.gender(this)))   // 가상 인물을 성별·연령대가 맞는 사람으로 고르는 데만
                        .put("face_check", faceCheckInfo())
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

    /** 녹화 중 화면을 떠나면 (홈 버튼·다른 앱) 거기까지 저장하고 멈춘다. */
    @Override
    protected void onPause() {
        super.onPause();
        if (recording && !busy && !isChangingConfigurations()) stop();
        if (countdownLeft > 0) cancelCountdown();
    }

    @Override
    protected void onDestroy() {
        ui.removeCallbacksAndMessages(null);
        if (recording) { try { camera.stop(dir); beacon.stop(); audio.stop(dir); imu.stop(); sync.stop(); } catch (Exception ignored) {} }
        if (camera != null) camera.close();
        faceWorker.shutdownNow();
        super.onDestroy();
    }

    /** 얼굴 자리 동그라미 (바깥은 어둡게, 정면 얼굴이면 초록 테두리). */
    static final class FaceOval extends View {
        private final Paint dim = new Paint(), line = new Paint(Paint.ANTI_ALIAS_FLAG), clear = new Paint(Paint.ANTI_ALIAS_FLAG);
        private final RectF r = new RectF();
        private boolean ok;

        FaceOval(Context c) {
            super(c);
            setLayerType(LAYER_TYPE_SOFTWARE, null);
            dim.setColor(0x77000000);
            clear.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.CLEAR));
            line.setStyle(Paint.Style.STROKE);
            line.setStrokeWidth(4 * c.getResources().getDisplayMetrics().density);
        }

        void setOk(boolean on) {
            ok = on;
            invalidate();
        }

        /** w×h 크기 그림에서 동그라미 자리 (화면과 같은 비율로). */
        RectF ovalIn(int w, int h) {
            float ow = Math.min(w * 0.62f, h * 0.42f), oh = ow * 1.3f;
            return new RectF((w - ow) / 2, h * 0.38f - oh / 2, (w + ow) / 2, h * 0.38f + oh / 2);
        }

        @Override
        protected void onDraw(Canvas c) {
            r.set(ovalIn(getWidth(), getHeight()));
            c.drawRect(0, 0, getWidth(), getHeight(), dim);
            c.drawOval(r, clear);
            line.setColor(ok ? Ui.OK : Color.WHITE);
            c.drawOval(r, line);
        }
    }
}
