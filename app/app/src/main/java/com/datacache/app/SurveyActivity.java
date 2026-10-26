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
 * 모두에게: 미션 결과(끝까지·일부만·못 함), 실수·다시 한 횟수. 생체 동의자에게: 힘듦·피로 1~5.
 * meta.json 의 outcome · self_report 에 넣고 결과 화면(업로드)으로.
 */
public class SurveyActivity extends Activity {
    private String outcome, retries;
    private int effort, fatigue;
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
            pick(b, "얼마나 힘들었어요?", scale(), v -> effort = Integer.parseInt(v));
            pick(b, "얼마나 피곤해요?", scale(), v -> fatigue = Integer.parseInt(v));
            Ui.add(this, body, b);
        }
        Ui.add(this, body, Ui.button(this, "보내기", true, v -> done(false)));
        Ui.add(this, body, Ui.button(this, "건너뛰기", false, v -> done(true)));
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
        try {
            File mf = new File(dir, "meta.json");
            JSONObject meta = new JSONObject(Files.read(mf));
            meta.put("outcome", new JSONObject().put("outcome", skipped || outcome == null ? JSONObject.NULL : outcome)
                    .put("retries", skipped || retries == null ? JSONObject.NULL : Integer.parseInt(retries)).put("skipped", skipped));
            if (Store.bioConsent(this)) {
                meta.put("self_report", new JSONObject().put("effort", skipped || effort == 0 ? JSONObject.NULL : effort)
                        .put("fatigue", skipped || fatigue == 0 ? JSONObject.NULL : fatigue));
            }
            Files.write(mf, meta.toString(2));
        } catch (Exception ignored) {
        }
        startActivity(new Intent(this, ResultActivity.class).putExtra("dir", dir.getAbsolutePath()).putExtra("upload", true));
        finish();
    }
}
