package com.datacash.app;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.widget.CheckBox;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;

/**
 * 동의: 영상(필수)과 생체(선택, 민감정보 — 따로). 연령대는 HRR 계산(최대 심박)과 다양성 관리에만 쓴다.
 * 철회는 설정에서 언제든 (서버에 알려 판매에서 뺀다).
 */
public class ConsentActivity extends BaseActivity {
    private String age = "";

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        LinearLayout body = Ui.page(this, getString(R.string.consent_title), false, Lang.button(this));
        LinearLayout c = Ui.card(this);
        c.addView(Ui.text(this, getString(R.string.consent_protect), 16, Ui.INK, true));
        TextView t = Ui.text(this, getString(R.string.consent_points), 14, Ui.SUB, false);
        t.setPadding(0, Ui.dp(this, 8), 0, 0);
        c.addView(t);
        Ui.add(this, body, c);

        LinearLayout c2 = Ui.card(this);
        CheckBox video = new CheckBox(this);
        video.setText(R.string.consent_video);
        video.setTextColor(Ui.INK);
        c2.addView(video);
        CheckBox bio = new CheckBox(this);
        bio.setText(R.string.consent_bio);
        bio.setTextColor(Ui.INK);
        c2.addView(bio);
        Ui.add(this, body, c2);

        LinearLayout c3 = Ui.card(this);
        c3.addView(Ui.text(this, getString(R.string.age_band), 15, Ui.INK, true));
        LinearLayout row = new LinearLayout(this);
        List<TextView> chips = new ArrayList<>();
        String[] ageVals = getResources().getStringArray(R.array.age_band_values);   // 서버 값
        String[] ageNames = getResources().getStringArray(R.array.age_bands);        // 보이는 이름
        for (int i = 0; i < ageVals.length; i++) {
            String a = ageVals[i];
            TextView ch = Ui.chip(this, ageNames[i], Ui.SUB, Ui.BG);
            ch.setOnClickListener(v -> {
                age = a;
                for (TextView x : chips) {
                    boolean on = x == ch;
                    x.setTextColor(on ? Ui.PRIMARY_DARK : Ui.SUB);
                    x.setBackground(Ui.round(on ? Ui.PRIMARY_SOFT : Ui.BG, Ui.dp(this, 99)));
                }
            });
            chips.add(ch);
            LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(-2, -2);
            lp.rightMargin = Ui.dp(this, 6);
            lp.topMargin = Ui.dp(this, 8);
            row.addView(ch, lp);
        }
        c3.addView(row);
        TextView gl = Ui.text(this, getString(R.string.gender_label), 15, Ui.INK, true);
        gl.setPadding(0, Ui.dp(this, 16), 0, 0);
        c3.addView(gl);
        LinearLayout grow = new LinearLayout(this);
        List<TextView> gchips = new ArrayList<>();
        String[][] gs = {{"female", getString(R.string.gender_female)}, {"male", getString(R.string.gender_male)}, {"", getString(R.string.gender_none)}};
        for (String[] g : gs) {
            TextView ch = Ui.chip(this, g[1], Ui.SUB, Ui.BG);
            ch.setOnClickListener(v -> {
                Store.setGender(this, g[0]);
                for (TextView x : gchips) {
                    boolean on = x == ch;
                    x.setTextColor(on ? Ui.PRIMARY_DARK : Ui.SUB);
                    x.setBackground(Ui.round(on ? Ui.PRIMARY_SOFT : Ui.BG, Ui.dp(this, 99)));
                }
            });
            gchips.add(ch);
            LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(-2, -2);
            lp.rightMargin = Ui.dp(this, 6);
            lp.topMargin = Ui.dp(this, 8);
            grow.addView(ch, lp);
        }
        c3.addView(grow);
        Ui.add(this, body, c3);

        Ui.add(this, body, Ui.button(this, getString(R.string.consent_start), true, v -> {
            if (!video.isChecked() || age.isEmpty()) {
                Toast.makeText(this, R.string.consent_need, Toast.LENGTH_SHORT).show();
                return;
            }
            Store.setConsent(this, true, bio.isChecked(), age);
            boolean wantBio = bio.isChecked();
            String self = getResources().getStringArray(R.array.relation_values)[0];   // 서버 관계 값 '본인'
            String myName = getString(R.string.me_name);
            new Thread(() -> {   // 본인을 동의 인원 명부에 올리고 바로 얼굴 등록으로 (서버가 없으면 나중에 '개인정보 동의 인원'에서)
                try {
                    JSONObject r = Uploader.callJson(this, "GET", "/api/people", null);
                    boolean hasMe = false;
                    org.json.JSONArray ps = r.optJSONArray("people");
                    for (int i = 0; ps != null && i < ps.length(); i++) hasMe |= ps.optJSONObject(i).optString("relation").equals(self);
                    if (!hasMe) {
                        JSONObject me = Uploader.callJson(this, "POST", "/api/people", new JSONObject().put("name", myName).put("relation", self)
                                .put("adult", true).put("age_band", age).put("agreed", true).put("bio", wantBio));
                        startActivity(new Intent(this, FaceActivity.class).putExtra("pid", me.optString("id")).putExtra("name", myName));
                    }
                } catch (Exception ignored) {
                }
                runOnUiThread(this::finish);
            }).start();
        }));
    }
}
