package com.datacache.app;

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
 * 미션 후 질문 — 본인이 답한 것만 라벨로 쓴다 (영상으로 추정하지 않는다).
 * 모두에게: 미션 결과(끝까지·일부만·못 함), 실수·다시 한 횟수.
 * 생체 동의자에게: 기분·피로·힘듦 1~5 (끝 낱말 표시), 지금 몸 상태 여러 개 고르기(기타는 글로). 뒤로 가기는 건너뛰기.
 * meta.json 의 outcome · self_report 에 넣고 결과 화면(업로드)으로.
 */
public class SurveyActivity extends Activity {
    private String outcome, retries;
    private int effort, fatigue, mood;
    private final java.util.Set<String> feelings = new java.util.LinkedHashSet<>();
    private android.widget.EditText other;
    private boolean sent;
    private File dir;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        dir = new File(getIntent().getStringExtra("dir"));
        LinearLayout body = Ui.page(this, "미션 후 한마디", false);
        LinearLayout c = Ui.card(this);
        c.addView(Ui.text(this, "로봇은 실수하고 다시 하는 장면에서도 배워요. 있는 그대로 골라 주세요. 건너뛰어도 보상은 같아요.", 14, Ui.SUB, false));
        pick(c, "미션은 어떻게 됐어요?", new String[][]{{"success", "끝까지 해냈어요"}, {"partial", "일부만 했어요"}, {"failed", "못 했어요"}}, v -> outcome = v);
        pick(c, "실수하거나 다시 한 적 있어요?", new String[][]{{"0", "없어요"}, {"1", "1번"}, {"2", "2번 이상"}}, v -> retries = v);
        Ui.add(this, body, c);
        if (Store.bioConsent(this)) {
            LinearLayout b = Ui.card(this);
            b.addView(Ui.text(this, "몸은 어땠어요? (심박·체온 데이터의 정답이 돼요)", 14, Ui.SUB, false));
            pick(b, "지금 기분은 어때요?", scale(), v -> mood = Integer.parseInt(v));
            ends(b, "매우 나쁨", "매우 좋음");
            pick(b, "얼마나 피곤해요?", scale(), v -> fatigue = Integer.parseInt(v));
            ends(b, "전혀", "매우 피곤");
            pick(b, "이번 미션은 얼마나 힘들었어요?", scale(), v -> effort = Integer.parseInt(v));
            ends(b, "아주 쉬움", "아주 힘듦");
            TextView q = Ui.text(this, "지금 몸은 어때요? (여러 개 골라도 돼요)", 16, Ui.INK, true);
            q.setPadding(0, Ui.dp(this, 14), 0, Ui.dp(this, 8));
            b.addView(q);
            other = Ui.input(this, "기타 (적어 주세요)", "", 0);
            other.setVisibility(android.view.View.GONE);
            LinearLayout grid = new LinearLayout(this);
            grid.setOrientation(LinearLayout.VERTICAL);
            String[] opts = {"숨이 차요", "땀이 나요", "손이 차가워요", "근육이 뻐근해요", "기타"};
            LinearLayout line = null;
            for (int i = 0; i < opts.length; i++) {
                if (i % 3 == 0) { line = new LinearLayout(this); grid.addView(line, Ui.margins(this, -1, -2, 0, 0, 0, 0, 6)); }
                String o = opts[i];
                TextView ch = Ui.text(this, o, 14, Ui.SUB, true);
                ch.setGravity(Gravity.CENTER);
                ch.setPadding(0, Ui.dp(this, 11), 0, Ui.dp(this, 11));
                ch.setBackground(Ui.round(Ui.FIELD, Ui.dp(this, 12)));
                ch.setOnClickListener(v -> {
                    boolean on = !feelings.remove(o);
                    if (on) feelings.add(o);
                    ch.setTextColor(on ? android.graphics.Color.WHITE : Ui.SUB);
                    ch.setBackground(Ui.round(on ? Ui.PRIMARY : Ui.FIELD, Ui.dp(this, 12)));
                    if (o.equals("기타")) other.setVisibility(on ? android.view.View.VISIBLE : android.view.View.GONE);
                });
                line.addView(ch, Ui.margins(this, 0, -2, 1f, 0, 0, 6, 0));
            }
            if (line != null) for (int k = opts.length % 3; k > 0 && k < 3; k++) line.addView(new android.view.View(this), Ui.margins(this, 0, 1, 1f, 0, 0, 6, 0));
            b.addView(grid);
            b.addView(other, Ui.margins(this, -1, -2, 0, 0, 6, 0, 0));
            Ui.add(this, body, b);
        }
        Ui.add(this, body, Ui.button(this, "보내기", true, v -> done(false)));
        Ui.add(this, body, Ui.button(this, "건너뛰기", false, v -> done(true)));
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
            if (Store.bioConsent(this)) {
                org.json.JSONArray feel = new org.json.JSONArray();
                for (String x : feelings) feel.put(x.equals("기타") && other != null ? "기타: " + other.getText().toString().trim() : x);
                meta.put("self_report", new JSONObject().put("effort", skipped || effort == 0 ? JSONObject.NULL : effort)
                        .put("fatigue", skipped || fatigue == 0 ? JSONObject.NULL : fatigue)
                        .put("mood", skipped || mood == 0 ? JSONObject.NULL : mood)
                        .put("body", skipped ? new org.json.JSONArray() : feel).put("version", "survey-v0.2"));
            }
            Files.write(mf, meta.toString(2));
        } catch (Exception ignored) {
        }
        Uploader.save(dir, "state", "saved");
        ProcessService.enqueue(this, dir);          // 앱을 닫아도 올리고 검수 결과를 알려 준다
        android.widget.Toast.makeText(this, "저장됐어요 · 검수하러 올리는 중이에요", android.widget.Toast.LENGTH_LONG).show();
        startActivity(new Intent(this, HomeActivity.class).addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP | Intent.FLAG_ACTIVITY_SINGLE_TOP));
        finish();
    }
}
