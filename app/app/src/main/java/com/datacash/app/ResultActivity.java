package com.datacash.app;

import android.app.Activity;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.util.Locale;

/** 올리기 → 서버 처리(가리기·라벨·4감 동기화) → 승인과 적립. 결과에 동기화 방법·검증된 단계 수를 보여 준다. */
public class ResultActivity extends BaseActivity {
    private File dir;
    private TextView state, detail;
    private ProgressBar bar;
    private final Handler ui = new Handler(Looper.getMainLooper());

    @Override
    protected void onCreate(Bundle state0) {
        super.onCreate(state0);
        dir = new File(getIntent().getStringExtra("dir"));
        LinearLayout body = Ui.page(this, getString(R.string.result_title), true);
        LinearLayout c = Ui.card(this);
        JSONObject local = Uploader.localState(dir);
        c.addView(Ui.text(this, Store.titleOf(this, local, dir.getName()), 18, Ui.INK, true));
        c.addView(Ui.text(this, dir.getName(), 12, Ui.SUB, false));
        state = Ui.text(this, Uploader.stateLabel(this, local.optString("state", "saved")), 22, Ui.PRIMARY_DARK, true);
        state.setPadding(0, Ui.dp(this, 14), 0, Ui.dp(this, 6));
        c.addView(state);
        bar = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        bar.setMax(100);
        bar.setProgressTintList(android.content.res.ColorStateList.valueOf(Ui.PRIMARY));
        c.addView(bar);
        detail = Ui.text(this, "", 14, Ui.SUB, false);
        detail.setPadding(0, Ui.dp(this, 10), 0, 0);
        c.addView(detail);
        Ui.add(this, body, c);
        Ui.add(this, body, Ui.button(this, getString(R.string.upload_again), false, v -> upload()));
        if (getIntent().getBooleanExtra("upload", false)) upload(); else poll();
    }

    private void upload() {
        new Thread(() -> {
            try {
                Uploader.upload(this, dir, (step, pct) -> ui.post(() -> { state.setText(getString(R.string.progress_fmt, step, pct)); bar.setProgress(pct); }));
                ui.post(this::poll);
            } catch (Exception e) {
                Uploader.save(dir, "state", "upload_failed");
                ui.post(() -> { state.setText(R.string.upload_failed_short); detail.setText(getString(R.string.upload_failed_detail, e.getMessage())); });
            }
        }).start();
    }

    private void poll() {
        new Thread(() -> {
            try {
                JSONObject st = Uploader.status(this, dir);
                ui.post(() -> show(st));
                String s = st.optString("state");
                if (s.equals("queued") || s.equals("processing") || s.equals("uploading")) ui.postDelayed(this::poll, 3000);
            } catch (Exception e) {
                ui.post(() -> detail.setText(getString(R.string.status_fail, e.getMessage())));
            }
        }).start();
    }

    private void show(JSONObject st) {
        String s = st.optString("state");
        state.setText(Uploader.stateLabel(this, s));
        bar.setProgress(s.equals("done") || s.equals("collected") ? 100 : s.equals("processing") ? 60 : s.equals("queued") ? 40 : bar.getProgress());
        StringBuilder b = new StringBuilder();
        if (s.equals("processing") || s.equals("queued")) b.append(getString(R.string.result_processing));
        if (s.equals("collected")) b.append(getString(R.string.result_collected));
        if (s.equals("done") && st.optJSONObject("calibration") != null) {      // 쥐는 세기 맞추기
            JSONObject cal = st.optJSONObject("calibration");
            state.setText(R.string.calib_learned);
            detail.setText(getString(R.string.calib_detail,
                    cal.optInt("n_windows"), cal.optDouble("accuracy_cv", 0) * 100));
            bar.setProgress(100);
            return;
        }
        if (s.equals("done")) {
            int r = st.optInt("reward");
            state.setText(getString(R.string.reward_added, r));
            b.append(getString(R.string.result_detail, st.optDouble("minutes"), st.optString("level"), st.optString("sync")));
            JSONArray sk = st.optJSONArray("skipped");
            if (sk != null && sk.length() > 0) b.append("\n").append(getString(R.string.skipped_steps, sk.toString()));
            Payout.checkNew(this);
        }
        if (s.equals("rejected") || s.equals("failed")) b.append(st.optString("error"));
        detail.setText(b.toString());
    }
}
