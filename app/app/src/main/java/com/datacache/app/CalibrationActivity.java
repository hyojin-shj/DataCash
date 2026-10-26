package com.datacache.app;

import android.app.Activity;
import android.content.Intent;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.text.InputType;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.util.Locale;

/**
 * 쥐는 힘 보정 (한 번만): 워치를 차고 손목 쪽 손으로 주방 저울을 세 번(가볍게·보통·세게) 2초 이상 누른다.
 * 누르는 동안 워치가 진동 펄스를 보내고 반응을 기록한다 (서버 /api/live 로 같은 세션 id).
 * 누른 시각(폰 시계)과 저울 숫자(kg)를 meta.calibration 에 적어 올리면, 서버가 워치 시계로 옮겨
 * 이 사람·이 워치의 보정을 저장하고 다음 녹화부터 쥐는 힘(N)을 계산한다.
 */
public class CalibrationActivity extends Activity {
    private static final String[] STEPS = {"가볍게 (1kg 쯤)", "보통 (3kg 쯤)", "세게 (5kg 이상)"};
    private final Handler ui = new Handler(Looper.getMainLooper());
    private final JSONArray presses = new JSONArray();
    private ServerSync sync;
    private File dir;
    private long pressStart;
    private TextView guide;
    private EditText kg;
    private TextView action;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        LinearLayout body = Ui.page(this, "쥐는 힘 맞추기", true);
        LinearLayout c = Ui.card(this);
        c.addView(Ui.text(this, "워치를 차고, 워치 쪽 손으로 주방 저울을 눌러 주세요. 한 번만 하면 돼요.", 15, Ui.INK, false));
        guide = Ui.text(this, "", 18, Ui.PRIMARY, true);
        guide.setPadding(0, Ui.dp(this, 12), 0, Ui.dp(this, 6));
        c.addView(guide);
        kg = new EditText(this);
        kg.setHint("저울 숫자 (kg)");
        kg.setInputType(InputType.TYPE_CLASS_NUMBER | InputType.TYPE_NUMBER_FLAG_DECIMAL);
        kg.setEnabled(false);
        c.addView(kg);
        Ui.add(this, body, c);
        action = Ui.button(this, "누르기 시작", true, v -> next());
        Ui.add(this, body, action);
        dir = new File(Store.sessionsDir(this), Clock.sessionId());
        dir.mkdirs();
        sync = new ServerSync(this);
        try {
            sync.start(dir);
        } catch (Exception e) {
            Toast.makeText(this, "서버에 연결 안 됨: " + e.getMessage(), Toast.LENGTH_LONG).show();
        }
        show();
    }

    private void show() {
        int k = presses.length();
        guide.setText(k < STEPS.length ? String.format(Locale.KOREA, "%d/%d · %s", k + 1, STEPS.length, STEPS[k]) : "다 했어요");
    }

    private void next() {
        if (pressStart == 0) {
            pressStart = Clock.now();
            action.setText("저울 숫자 적고 손 떼기");
            kg.setEnabled(true);
            kg.requestFocus();
            return;
        }
        double w;
        try {
            w = Double.parseDouble(kg.getText().toString().trim());
        } catch (NumberFormatException e) {
            Toast.makeText(this, "누르는 동안 저울 숫자를 적어 주세요", Toast.LENGTH_SHORT).show();
            return;
        }
        long end = Clock.now();
        if (end - pressStart < 2_000_000_000L) {
            Toast.makeText(this, "2초 이상 눌러 주세요", Toast.LENGTH_SHORT).show();
            return;
        }
        try {
            presses.put(new JSONObject().put("start_ns", pressStart).put("end_ns", end).put("force_n", w * 9.81));
        } catch (Exception ignored) {
        }
        pressStart = 0;
        kg.setText("");
        kg.setEnabled(false);
        if (presses.length() < STEPS.length) {
            action.setText("누르기 시작");
            show();
            return;
        }
        show();
        action.setText("올리는 중…");
        action.setEnabled(false);
        new Thread(this::finishAndUpload, "calibration").start();
    }

    private void finishAndUpload() {
        try {
            Thread.sleep(1500);     // 손 뗀 뒤 진동 반응(0N)도 조금 담는다
            sync.stop();
            JSONObject meta = new JSONObject()
                    .put("session_id", dir.getName()).put("format", "datacache/1").put("clock", "BOOTTIME")
                    .put("mission", new JSONObject().put("id", "grip_calibration").put("title", "쥐는 힘 맞추기"))
                    .put("devices", new JSONObject().put("phone", new JSONObject().put("model", Build.MANUFACTURER + " " + Build.MODEL)
                            .put("android", Build.VERSION.RELEASE).put("id", Store.deviceId(this))))
                    .put("participant", new JSONObject().put("consent", Store.consentJson(this)).put("age_band", Store.ageBand(this)))
                    .put("calibration", new JSONObject().put("clock", "phone").put("presses", presses))
                    .put("recorded_at", Clock.wallIso());
            Files.write(new File(dir, "meta.json"), meta.toString(2));
            Uploader.save(dir, "title", "쥐는 힘 맞추기");
            Uploader.upload(this, dir, (s, p) -> ui.post(() -> action.setText(s + " " + p + "%")));
            ui.post(() -> {
                startActivity(new Intent(this, ResultActivity.class).putExtra("dir", dir.getAbsolutePath()));
                finish();
            });
        } catch (Exception e) {
            ui.post(() -> {
                action.setEnabled(true);
                action.setText("다시 올리기");
                action.setOnClickListener(v -> new Thread(this::finishAndUpload, "calibration").start());
                Toast.makeText(this, "올리기 실패: " + e.getMessage(), Toast.LENGTH_LONG).show();
            });
        }
    }
}
