package com.datacache.app;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Color;
import android.os.Bundle;
import android.view.Gravity;
import android.widget.LinearLayout;
import android.widget.TextView;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

/** 홈: 적립금, 미션 목록(부족한 미션 보너스), 내 녹화 기록. 처음이면 동의 화면부터. */
public class HomeActivity extends Activity {
    private LinearLayout body;
    private final Map<String, Integer> bonus = new HashMap<>();

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        if (!Store.consented(this)) startActivity(new Intent(this, ConsentActivity.class));
    }

    @Override
    protected void onResume() {
        super.onResume();
        render();
        new Thread(() -> {   // 서버가 알려 주는 부족한 미션 보너스 (없으면 보너스 표시 없이)
            try {
                JSONArray ms = Uploader.missions(this);
                Map<String, Integer> b = new HashMap<>();
                for (int i = 0; i < ms.length(); i++) {
                    JSONObject m = ms.getJSONObject(i);
                    if (m.optInt("bonus_pct") > 0) b.put(m.getString("id"), m.getInt("bonus_pct"));
                }
                runOnUiThread(() -> { bonus.clear(); bonus.putAll(b); render(); });
            } catch (Exception ignored) {
            }
        }).start();
    }

    private void render() {
        body = Ui.page(this, "데이터캐시", false);
        LinearLayout hero = new LinearLayout(this);
        hero.setOrientation(LinearLayout.VERTICAL);
        hero.setBackground(Ui.gradient(Ui.dp(this, 22)));
        hero.setPadding(Ui.dp(this, 20), Ui.dp(this, 18), Ui.dp(this, 20), Ui.dp(this, 20));
        hero.addView(Ui.text(this, "모은 적립금", 14, Color.WHITE, false));
        hero.addView(Ui.text(this, String.format(Locale.KOREA, "%,d원", Store.earned(this)), 32, Color.WHITE, true));
        TextView sub = Ui.text(this, "집에서 미션을 찍으면 로봇이 배울 4감 데이터가 돼요.\n얼굴은 실존하지 않는 가상 인물로 바뀌고, 원본은 판매되지 않아요.", 13, Color.WHITE, false);
        sub.setPadding(0, Ui.dp(this, 8), 0, 0);
        hero.addView(sub);
        Ui.add(this, body, hero);

        LinearLayout senses = new LinearLayout(this);
        for (String[] s : new String[][]{{"촉각", "#FF5F5F"}, {"시각", "#2E9BF0"}, {"청각", "#D97F1F"}, {"몸 감각", "#8E5CF7"}}) {
            TextView c = Ui.chip(this, s[0], Color.parseColor(s[1]), Color.WHITE);
            c.setGravity(Gravity.CENTER);
            LinearLayout.LayoutParams lp = Ui.weight();
            lp.rightMargin = Ui.dp(this, 6);
            senses.addView(c, lp);
        }
        Ui.add(this, body, senses);

        body.addView(Ui.text(this, "미션", 17, Ui.INK, true));
        try {
            JSONArray ms = Store.missions(this);
            for (int i = 0; i < ms.length(); i++) {
                JSONObject m = ms.getJSONObject(i);
                LinearLayout c = Ui.card(this);
                LinearLayout top = new LinearLayout(this);
                top.setGravity(Gravity.CENTER_VERTICAL);
                top.addView(Ui.text(this, m.getString("title"), 17, Ui.INK, true), Ui.weight());
                int per = Store.bioConsent(this) ? 833 : 550;
                top.addView(Ui.chip(this, String.format(Locale.KOREA, "분당 %,d원", per), Ui.PRIMARY_DARK, Ui.PRIMARY_SOFT));
                Integer b = bonus.get(m.getString("id"));
                if (b != null) {
                    TextView bc = Ui.chip(this, "부족 +" + b + "%", Ui.ACCENT_INK, Ui.ACCENT_SOFT);
                    LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(-2, -2);
                    lp.leftMargin = Ui.dp(this, 6);
                    top.addView(bc, lp);
                }
                c.addView(top);
                TextView g = Ui.text(this, m.getString("guide"), 13, Ui.SUB, false);
                g.setPadding(0, Ui.dp(this, 6), 0, 0);
                c.addView(g);
                String id = m.getString("id");
                c.setOnClickListener(v -> startActivity(new Intent(this, MissionActivity.class).putExtra("mission", id)));
                Ui.add(this, body, c);
            }
        } catch (Exception e) {
            body.addView(Ui.text(this, "미션을 읽지 못했어요: " + e.getMessage(), 13, Ui.SUB, false));
        }

        File[] ss = Store.sessionsDir(this).listFiles();
        if (ss != null && ss.length > 0) {
            body.addView(Ui.text(this, "내 녹화", 17, Ui.INK, true));
            Arrays.sort(ss, (x, y) -> y.getName().compareTo(x.getName()));
            for (int i = 0; i < Math.min(6, ss.length); i++) {
                File d = ss[i];
                JSONObject st = Uploader.localState(d);
                LinearLayout c = Ui.card(this);
                LinearLayout top = new LinearLayout(this);
                top.setGravity(Gravity.CENTER_VERTICAL);
                top.addView(Ui.text(this, st.optString("title", d.getName()), 15, Ui.INK, true), Ui.weight());
                String state = st.optString("state", "saved");
                int reward = st.optInt("reward", 0);
                top.addView(reward > 0 ? Ui.chip(this, String.format(Locale.KOREA, "+%,d원", reward), Ui.OK, Ui.OK_SOFT)
                        : Ui.chip(this, Uploader.stateKo(state), Ui.SUB, Ui.BG));
                c.addView(top);
                c.addView(Ui.text(this, d.getName(), 12, Ui.SUB, false));
                c.setOnClickListener(v -> startActivity(new Intent(this, ResultActivity.class).putExtra("dir", d.getAbsolutePath())));
                Ui.add(this, body, c);
            }
        }
        TextView cal = Ui.text(this, "⌚ 워치로 쥐는 힘 맞추기 (한 번만)", 14, Ui.ACCENT_INK, true);
        cal.setGravity(Gravity.CENTER);
        cal.setPadding(0, Ui.dp(this, 14), 0, 0);
        cal.setOnClickListener(v -> startActivity(new Intent(this, CalibrationActivity.class)));
        body.addView(cal);
        TextView set = Ui.text(this, "설정 · 서버 연결", 14, Ui.SUB, false);
        set.setGravity(Gravity.CENTER);
        set.setPadding(0, Ui.dp(this, 10), 0, 0);
        set.setOnClickListener(v -> startActivity(new Intent(this, SettingsActivity.class)));
        body.addView(set);
    }
}
