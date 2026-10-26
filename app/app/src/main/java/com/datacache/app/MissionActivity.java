package com.datacache.app;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.view.Gravity;
import android.widget.LinearLayout;
import android.widget.TextView;

import org.json.JSONArray;
import org.json.JSONObject;

/** 미션 상세: 목적, 촬영 안내, 상황 10가지(조건을 하나씩 바꿔 다양성을 채운다). 상황을 누르면 녹화. */
public class MissionActivity extends Activity {
    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        String id = getIntent().getStringExtra("mission");
        try {
            JSONObject m = Store.mission(this, id);
            LinearLayout body = Ui.page(this, m.getString("title"), true);
            LinearLayout c = Ui.card(this);
            c.addView(Ui.text(this, m.getString("guide"), 15, Ui.INK, true));
            TextView p = Ui.text(this, "로봇이 배우는 것 · " + m.getString("purpose"), 13, Ui.SUB, false);
            p.setPadding(0, Ui.dp(this, 6), 0, 0);
            c.addView(p);
            JSONArray prims = m.optJSONArray("primitives");
            if (prims != null) {
                JSONObject ko = Store.primitivesKo(this);
                StringBuilder seq = new StringBuilder();
                for (int i = 0; i < prims.length(); i++) {
                    if (i > 0) seq.append(" → ");
                    seq.append(ko.optString(prims.getString(i), prims.getString(i)));
                }
                TextView q = Ui.text(this, "동작 순서 · " + seq, 13, Ui.PRIMARY_DARK, true);
                q.setPadding(0, Ui.dp(this, 6), 0, 0);
                c.addView(q);
            }
            TextView tip = Ui.text(this, "워치를 차고 찍으면 촉각(쥐는 힘)·몸 감각까지 담겨 보상이 더 커요. 녹화 중 실수하면 '실수했어요'를 눌러 주세요 — 로봇은 실수에서도 배워요.",
                    13, Ui.ACCENT_INK, false);
            tip.setBackground(Ui.round(Ui.ACCENT_SOFT, Ui.dp(this, 12)));
            tip.setPadding(Ui.dp(this, 12), Ui.dp(this, 10), Ui.dp(this, 12), Ui.dp(this, 10));
            LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(-1, -2);
            lp.topMargin = Ui.dp(this, 12);
            c.addView(tip, lp);
            Ui.add(this, body, c);
            body.addView(Ui.text(this, "찍을 상황 고르기", 16, Ui.INK, true));
            JSONArray sits = m.getJSONArray("situations");
            for (int i = 0; i < sits.length(); i++) {
                String s = sits.getString(i);
                LinearLayout row = Ui.card(this);
                row.setOrientation(LinearLayout.HORIZONTAL);
                row.setGravity(Gravity.CENTER_VERTICAL);
                row.addView(Ui.text(this, (i + 1) + ".  " + s, 15, Ui.INK, false), Ui.weight());
                row.addView(Ui.chip(this, "찍기", Ui.PRIMARY_DARK, Ui.PRIMARY_SOFT));
                row.setOnClickListener(v -> startActivity(new Intent(this, RecordActivity.class)
                        .putExtra("mission", id).putExtra("situation", s)));
                Ui.add(this, body, row);
            }
        } catch (Exception e) {
            Ui.page(this, "미션", true).addView(Ui.text(this, "미션을 읽지 못했어요", 14, Ui.SUB, false));
        }
    }
}
