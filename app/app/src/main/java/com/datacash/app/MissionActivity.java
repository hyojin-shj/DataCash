package com.datacash.app;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.view.Gravity;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.TextView;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.Locale;
import java.util.Set;

/** 미션 상세: 머리 카드(보상·의뢰 목적·동작 순서·찍은 상황 수) + 찍을 상황 목록. 상황을 누르면 촬영 화면. */
public class MissionActivity extends BaseActivity {
    private String id;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        id = getIntent().getStringExtra("mission");
    }

    @Override
    protected void onResume() {
        super.onResume();
        try {
            render(Store.mission(this, id));
        } catch (Exception e) {
            Ui.page(this, getString(R.string.mission), true).addView(Ui.text(this, getString(R.string.mission_read_fail), 14, Ui.SUB, false));
        }
    }

    private void render(JSONObject m) throws Exception {
        LinearLayout body = Ui.page(this, m.getString("title"), true);
        JSONArray sits = m.getJSONArray("situations");
        Set<String> done = Store.doneSituations(this, id);
        boolean today = id.equals(Store.todayMission(this));

        LinearLayout c = Ui.card(this);
        LinearLayout head = new LinearLayout(this);
        head.setGravity(Gravity.CENTER_VERTICAL);
        FrameLayout tile = Ui.iconTile(this, Icon.TARGET, Ui.PRIMARY, Ui.PRIMARY_SOFT, 52);
        head.addView(tile, Ui.margins(this, Ui.dp(this, 52), Ui.dp(this, 52), 0, 0, 0, 14, 0));
        LinearLayout words = new LinearLayout(this);
        words.setOrientation(LinearLayout.VERTICAL);
        words.addView(Ui.text(this, m.getString("title"), 20, Ui.INK, true));
        String sponsor = m.optString("sponsor");
        TextView purpose = Ui.text(this, (sponsor.isEmpty() ? "" : getString(R.string.sponsor_line, sponsor) + "\n") + getString(R.string.purpose_line, m.getString("purpose")), 13, Ui.SUB, false);
        purpose.setPadding(0, Ui.dp(this, 5), 0, 0);
        words.addView(purpose);
        head.addView(words, Ui.weight());
        c.addView(head);

        int per = Store.perMission(this);
        LinearLayout rw = new LinearLayout(this);
        rw.setGravity(Gravity.CENTER_VERTICAL);
        rw.setPadding(0, Ui.dp(this, 18), 0, Ui.dp(this, 8));
        rw.addView(Ui.text(this, Ui.plusWon(this, per), 26, Ui.PRIMARY, true));
        if (today) {
            TextView b = Ui.chip(this, getString(R.string.today_bonus_amount, per / 2), Ui.ACCENT_INK, Ui.ACCENT_SOFT);
            rw.addView(b, Ui.margins(this, -2, -2, 0, 10, 0, 0, 0));
        }
        c.addView(rw);
        c.addView(Ui.text(this, getString(R.string.per_minute, m.getString("guide")), 14, Ui.SUB, false));

        JSONArray prims = m.optJSONArray("primitives");
        if (prims != null) {
            JSONObject ko = Store.lang(this).equals("ko") ? Store.primitivesKo(this) : new JSONObject();   // 다른 언어는 동작 이름 그대로 (reach, grasp…)
            StringBuilder seq = new StringBuilder();
            for (int i = 0; i < prims.length(); i++) {
                if (i > 0) seq.append(" → ");
                seq.append(ko.optString(prims.getString(i), prims.getString(i)));
            }
            TextView q = Ui.text(this, getString(R.string.steps_seq, seq.toString()), 13, Ui.PRIMARY_DARK, true);
            q.setPadding(0, Ui.dp(this, 10), 0, 0);
            c.addView(q);
        }
        TextView n = Ui.text(this, getString(R.string.situations_done, done.size(), sits.length()), 13, Ui.INK, true);
        n.setPadding(0, Ui.dp(this, 16), 0, Ui.dp(this, 8));
        c.addView(n);
        c.addView(Ui.progress(this, sits.length() == 0 ? 0f : (float) done.size() / sits.length()),
                new LinearLayout.LayoutParams(-1, Ui.dp(this, 6)));
        TextView tip = Ui.text(this, getString(R.string.mission_tip),
                13, Ui.ACCENT_INK, false);
        tip.setBackground(Ui.round(Ui.ACCENT_SOFT, Ui.dp(this, 14)));
        tip.setPadding(Ui.dp(this, 14), Ui.dp(this, 12), Ui.dp(this, 14), Ui.dp(this, 12));
        c.addView(tip, Ui.margins(this, -1, -2, 0, 0, 16, 0, 0));
        Ui.add(this, body, c);

        body.addView(Ui.sectionTitle(this, getString(R.string.situations_pick_title)));
        LinearLayout list = Ui.listCard(this);
        for (int i = 0; i < sits.length(); i++) {
            String s = sits.getString(i);
            boolean d = done.contains(s);
            LinearLayout r = Ui.row(this, -1, 0, 0, Store.situationLabel(this, m, i), d ? getString(R.string.recorded_done) : null,
                    Ui.chip(this, getString(d ? R.string.again : R.string.record_chip), d ? Ui.SUB : Ui.PRIMARY_DARK, d ? Ui.FIELD : Ui.PRIMARY_SOFT),
                    v -> startActivity(new Intent(this, RecordActivity.class).putExtra("mission", id).putExtra("situation", s)));
            TextView num = Ui.text(this, d ? "✓" : String.valueOf(i + 1), 14, d ? Ui.OK : Ui.SUB, true);
            num.setGravity(Gravity.CENTER);
            num.setBackground(Ui.round(d ? Ui.OK_SOFT : Ui.FIELD, Ui.dp(this, 16)));
            r.addView(num, 0, Ui.margins(this, Ui.dp(this, 32), Ui.dp(this, 32), 0, 0, 0, 14, 0));
            list.addView(r);
        }
        Ui.add(this, body, list);
    }
}
