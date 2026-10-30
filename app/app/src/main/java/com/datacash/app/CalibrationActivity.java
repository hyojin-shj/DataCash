package com.datacash.app;

import android.app.Activity;
import android.content.Intent;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.Gravity;
import android.view.WindowManager;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.util.Locale;

/**
 * 쥐는 세기 맞추기 (한 번만, GripSense 방식 — Goel·Wobbrock·Patel, UIST 2012).
 * 워치를 찬 손으로 물병이나 컵을 쥐고, 화면 안내대로 '손 놓기 · 가볍게 · 평소처럼 · 꽉'을 8초씩 두 번 돌린다.
 * 저울 같은 기구 없이, 내가 스스로 구분한 세기로 배운다 (논문도 그렇게 학습).
 * 그동안 워치는 0.5초마다 진동하고 손목 자이로·가속도를 기록한다 (서버 /api/live 로 같은 세션).
 * 구간(폰 시계)과 단계를 meta.calibration 에 적어 올리면, 서버가 워치 시계로 옮겨 이 사람·이 워치의 결정 트리를 학습한다.
 */
public class CalibrationActivity extends BaseActivity {
    private String[] NAMES, HINTS;     // 손 놓기 · 가볍게 · 평소처럼 · 꽉 (R.array.calib_names · calib_hints)
    private static final int[] ORDER = {0, 1, 2, 3, 0, 1, 2, 3, 0};
    private static final long STEP_MS = 8000, GAP_MS = 2500;
    private final Handler ui = new Handler(Looper.getMainLooper());
    private final JSONArray segments = new JSONArray();
    private ServerSync sync;
    private File dir;
    private int idx = -1;
    private long stepStart;
    private TextView big, hint, count, progress;
    private TextView action;
    private boolean running;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        NAMES = getResources().getStringArray(R.array.calib_names);
        HINTS = getResources().getStringArray(R.array.calib_hints);
        LinearLayout body = Ui.page(this, getString(R.string.menu_calib), true);
        LinearLayout c = Ui.card(this);
        c.addView(Ui.text(this, getString(R.string.calib_intro),
                15, Ui.INK, false));
        TextView why = Ui.text(this, getString(R.string.calib_why),
                13, Ui.SUB, false);
        why.setPadding(0, Ui.dp(this, 8), 0, 0);
        c.addView(why);
        Ui.add(this, body, c);

        LinearLayout stage = Ui.card(this);
        stage.setGravity(Gravity.CENTER_HORIZONTAL);
        progress = Ui.text(this, "", 13, Ui.ACCENT_INK, true);
        stage.addView(progress, Ui.wrap());
        big = Ui.text(this, getString(R.string.calib_ready), 26, Ui.INK, true);
        big.setGravity(Gravity.CENTER);
        big.setPadding(0, Ui.dp(this, 10), 0, Ui.dp(this, 6));
        stage.addView(big, Ui.wrap());
        hint = Ui.text(this, getString(R.string.calib_open_watch), 14, Ui.SUB, false);
        hint.setGravity(Gravity.CENTER);
        stage.addView(hint, Ui.wrap());
        count = Ui.text(this, "", 56, Ui.PRIMARY, true);
        count.setPadding(0, Ui.dp(this, 10), 0, 0);
        stage.addView(count, Ui.wrap());
        Ui.add(this, body, stage);
        action = Ui.button(this, getString(R.string.start), true, v -> start());
        Ui.add(this, body, action);
    }

    private void start() {
        if (running) return;
        running = true;
        action.setEnabled(false);
        action.setText(R.string.calibrating);
        dir = new File(Store.sessionsDir(this), Clock.sessionId());
        dir.mkdirs();
        sync = new ServerSync(this);       // 녹화 시작을 서버에 알리면 워치가 같은 세션으로 기록·진동한다
        try {
            sync.start(dir);
        } catch (Exception e) {
            Toast.makeText(this, getString(R.string.server_fail_fmt, e.getMessage()), Toast.LENGTH_LONG).show();
        }
        big.setText(R.string.starting);
        hint.setText(R.string.wait_watch);
        ui.postDelayed(this::next, 3000);
    }

    /** 다음 단계: 2.5초 안내(구간에 넣지 않음) → 8초 구간. */
    private void next() {
        idx++;
        if (idx >= ORDER.length) { finishAndUpload(); return; }
        int lv = ORDER[idx];
        progress.setText(String.format(Locale.KOREA, "%d / %d", idx + 1, ORDER.length));
        big.setText(getString(R.string.next_step, NAMES[lv]));
        hint.setText(HINTS[lv]);
        count.setText("");
        ui.postDelayed(() -> {
            stepStart = Clock.now();
            big.setText(NAMES[lv]);
            tick(lv, STEP_MS / 1000);
        }, GAP_MS);
    }

    private void tick(int lv, long left) {
        if (left <= 0) {
            try {
                segments.put(new JSONObject().put("start_ns", stepStart).put("end_ns", Clock.now()).put("level", lv));
            } catch (Exception ignored) {
            }
            next();
            return;
        }
        count.setText(String.valueOf(left));
        ui.postDelayed(() -> tick(lv, left - 1), 1000);
    }

    private void finishAndUpload() {
        big.setText(R.string.finished);
        hint.setText(R.string.learning);
        count.setText("");
        new Thread(() -> {
            try {
                Thread.sleep(1000);
                sync.stop();
                JSONObject meta = new JSONObject()
                        .put("session_id", dir.getName()).put("format", "datacash/1").put("clock", "BOOTTIME")
                        .put("mission", new JSONObject().put("id", "grip_calibration").put("title", getString(R.string.menu_calib)))
                        .put("devices", new JSONObject().put("phone", new JSONObject().put("model", Build.MANUFACTURER + " " + Build.MODEL)
                                .put("android", Build.VERSION.RELEASE).put("id", Store.deviceId(this))))
                        .put("participant", new JSONObject().put("consent", Store.consentJson(this)).put("age_band", Store.ageBand(this)))
                        .put("calibration", new JSONObject().put("method", "gripsense").put("clock", "phone")
                                .put("levels", new JSONArray(java.util.Arrays.asList(getResources().getStringArray(R.array.grip_level_values)))).put("segments", segments))
                        .put("recorded_at", Clock.wallIso());
                Files.write(new File(dir, "meta.json"), meta.toString(2));
                Uploader.save(dir, "title", getString(R.string.menu_calib));
                Uploader.upload(this, dir, (s, p) -> ui.post(() -> hint.setText(getString(R.string.progress_fmt, s, p))));
                ui.post(() -> {
                    startActivity(new Intent(this, ResultActivity.class).putExtra("dir", dir.getAbsolutePath()));
                    finish();
                });
            } catch (Exception e) {
                ui.post(() -> {
                    action.setEnabled(true);
                    action.setText(R.string.upload_again);
                    action.setOnClickListener(v -> new Thread(this::retry, "calibration").start());
                    Toast.makeText(this, getString(R.string.upload_fail_fmt, Uploader.reason(e)), Toast.LENGTH_LONG).show();
                });
            }
        }, "calibration").start();
    }

    private void retry() {
        try {
            Uploader.upload(this, dir, (s, p) -> ui.post(() -> hint.setText(getString(R.string.progress_fmt, s, p))));
            ui.post(() -> { startActivity(new Intent(this, ResultActivity.class).putExtra("dir", dir.getAbsolutePath())); finish(); });
        } catch (Exception e) {
            ui.post(() -> Toast.makeText(this, getString(R.string.upload_fail_fmt, Uploader.reason(e)), Toast.LENGTH_LONG).show());
        }
    }

    @Override
    protected void onDestroy() {
        ui.removeCallbacksAndMessages(null);
        if (running && sync != null && idx < ORDER.length) { try { sync.stop(); } catch (Exception ignored) {} }
        super.onDestroy();
    }
}
