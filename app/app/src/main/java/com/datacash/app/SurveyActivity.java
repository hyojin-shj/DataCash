package com.datacash.app;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.view.Gravity;
import android.widget.LinearLayout;
import android.widget.TextView;

import org.json.JSONObject;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * 미션 후 질문 — 한 화면에서 1~2번 누르면 끝난다 (고르면 바로 저장·올리기). 본인이 답한 것만 라벨로 쓴다.
 * 모두에게: 미션 결과 하나 (한 번에 끝냄 · 다시 하고 끝냄 · 못 끝냄) → outcome·retries.
 * 생체 동의자에게만: 얼마나 힘들었는지 1~5 (심박·체온의 정답 라벨). 기분·피로·몸 상태는 뺐다 (점심 녹화에서 너무 많다는 의견).
 * 녹화 중 '실수 표시'는 따로 기록된다. 뒤로 가기는 건너뛰기.
 */
public class SurveyActivity extends BaseActivity {
    private String outcome, retries;
    private int effort;
    private boolean sent, bio;
    private File dir;
    private final android.os.Handler ui = new android.os.Handler(android.os.Looper.getMainLooper());

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        dir = new File(getIntent().getStringExtra("dir"));
        bio = Store.bioConsent(this);
        LinearLayout body = Ui.page(this, getString(R.string.survey_title), false);
        LinearLayout c = Ui.card(this);
        c.addView(Ui.text(this, getString(bio ? R.string.survey_hint_bio : R.string.survey_hint), 14, Ui.SUB, false));
        pick(c, getString(R.string.survey_q1), new String[][]{{"success:0", getString(R.string.survey_once)}, {"success:1", getString(R.string.survey_retry)},
                {"failed:", getString(R.string.survey_failed)}}, v -> {
            String[] p = v.split(":", -1);
            outcome = p[0];
            retries = p[1].isEmpty() ? null : p[1];
            maybeDone();
        });
        if (bio) {
            pick(c, getString(R.string.survey_q2), scale(), v -> { effort = Integer.parseInt(v); maybeDone(); });
            ends(c, getString(R.string.survey_easy), getString(R.string.survey_hard));
        }
        Ui.add(this, body, c);
        Ui.add(this, body, Ui.button(this, getString(R.string.skip), false, v -> done(true)));
    }

    /** 다 고르면 잠깐(고른 것이 보이게) 뒤 저장. */
    private void maybeDone() {
        if (outcome != null && (!bio || effort > 0)) ui.postDelayed(() -> done(false), 350);
    }

    private void ends(LinearLayout parent, String lo, String hi) {
        LinearLayout r = new LinearLayout(this);
        r.setPadding(Ui.dp(this, 2), Ui.dp(this, 6), Ui.dp(this, 8), 0);
        r.addView(Ui.text(this, lo, 12, Ui.SUB, false), Ui.weight());
        r.addView(Ui.text(this, hi, 12, Ui.SUB, false));
        parent.addView(r);
    }

    @Override
    public void onBackPressed() {
        done(true);   // 뒤로 가기 = 건너뛰기 (녹화는 그대로 올린다)
    }

    private static String[][] scale() {
        String[][] s = new String[5][2];
        for (int i = 0; i < 5; i++) s[i] = new String[]{String.valueOf(i + 1), String.valueOf(i + 1)};
        return s;
    }

    interface Pick { void on(String v); }

    private void pick(LinearLayout parent, String q, String[][] opts, Pick on) {
        TextView t = Ui.text(this, q, 16, Ui.INK, true);
        t.setPadding(0, Ui.dp(this, 14), 0, Ui.dp(this, 8));
        parent.addView(t);
        LinearLayout row = new LinearLayout(this);
        List<TextView> chips = new ArrayList<>();
        for (String[] o : opts) {
            TextView ch = Ui.text(this, o[1], 14, Ui.SUB, true);
            ch.setGravity(Gravity.CENTER);
            ch.setPadding(0, Ui.dp(this, 11), 0, Ui.dp(this, 11));
            ch.setBackground(Ui.round(Ui.BG, Ui.dp(this, 12)));
            ch.setOnClickListener(v -> {
                on.on(o[0]);
                for (TextView x : chips) {
                    boolean sel = x == ch;
                    x.setTextColor(sel ? android.graphics.Color.WHITE : Ui.SUB);
                    x.setBackground(Ui.round(sel ? Ui.PRIMARY : Ui.FIELD, Ui.dp(this, 12)));
                }
            });
            chips.add(ch);
            LinearLayout.LayoutParams lp = Ui.weight();
            lp.rightMargin = Ui.dp(this, 6);
            row.addView(ch, lp);
        }
        parent.addView(row);
    }

    private void done(boolean skipped) {
        if (sent) return;
        sent = true;
        try {
            File mf = new File(dir, "meta.json");
            JSONObject meta = new JSONObject(Files.read(mf));
            meta.put("outcome", new JSONObject().put("outcome", skipped || outcome == null ? JSONObject.NULL : outcome)
                    .put("retries", skipped || retries == null ? JSONObject.NULL : Integer.parseInt(retries)).put("skipped", skipped));
            if (bio) {
                meta.put("self_report", new JSONObject().put("effort", skipped || effort == 0 ? JSONObject.NULL : effort)
                        .put("version", "survey-v0.3"));
            }
            Files.write(mf, meta.toString(2));
        } catch (Exception ignored) {
        }
        Uploader.save(dir, "state", "saved");
        ProcessService.enqueue(this, dir);          // 앱을 닫아도 올리고 검수 결과를 알려 준다
        android.widget.Toast.makeText(this, R.string.survey_saved, android.widget.Toast.LENGTH_LONG).show();
        startActivity(new Intent(this, HomeActivity.class).addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_SINGLE_TOP));
        finish();
    }
}
