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
public class ConsentActivity extends Activity {
    private static final String[] AGES = {"19~29세", "30대", "40대", "50대", "60대 이상"};
    private String age = "";

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        LinearLayout body = Ui.page(this, "시작하기 전에", false);
        LinearLayout c = Ui.card(this);
        c.addView(Ui.text(this, "데이터캐시는 이렇게 지켜요", 16, Ui.INK, true));
        TextView t = Ui.text(this, "• 판매되는 영상에서 내 얼굴은 실존하지 않는 가상 인물로 바뀌어요\n"
                + "• 동의하지 않은 가족·화면·거울·액자·서류는 회색으로 가려요\n"
                + "• 이름·주소가 들린 말은 그 부분만 소리를 꺼요\n"
                + "• 동의를 철회하면 내 녹화는 판매에서 빠져요", 14, Ui.SUB, false);
        t.setPadding(0, Ui.dp(this, 8), 0, 0);
        c.addView(t);
        Ui.add(this, body, c);

        LinearLayout c2 = Ui.card(this);
        CheckBox video = new CheckBox(this);
        video.setText("(필수) 미션 영상·소리·움직임을 위 방식으로 처리해 로봇 학습 데이터로 판매하는 데 동의해요");
        video.setTextColor(Ui.INK);
        c2.addView(video);
        CheckBox bio = new CheckBox(this);
        bio.setText("(선택) 워치의 심박수·심박 예비량(HRR)·피부온도 변화와 미션 후 응답을 함께 판매하는 데 동의해요 — 건강을 판단하거나 조언하지 않아요");
        bio.setTextColor(Ui.INK);
        c2.addView(bio);
        Ui.add(this, body, c2);

        LinearLayout c3 = Ui.card(this);
        c3.addView(Ui.text(this, "연령대", 15, Ui.INK, true));
        LinearLayout row = new LinearLayout(this);
        List<TextView> chips = new ArrayList<>();
        for (String a : AGES) {
            TextView ch = Ui.chip(this, a, Ui.SUB, Ui.BG);
            ch.setOnClickListener(v -> {
                age = a;
                for (TextView x : chips) {
                    boolean on = x.getText().toString().equals(a);
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
        TextView gl = Ui.text(this, "성별 (선택 — 성별을 정한 의뢰 미션과, 얼굴을 바꿀 가상 인물 고르기에만 써요)", 15, Ui.INK, true);
        gl.setPadding(0, Ui.dp(this, 16), 0, 0);
        c3.addView(gl);
        LinearLayout grow = new LinearLayout(this);
        List<TextView> gchips = new ArrayList<>();
        String[][] gs = {{"female", "여성"}, {"male", "남성"}, {"", "밝히지 않음"}};
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

        Ui.add(this, body, Ui.button(this, "동의하고 시작하기", true, v -> {
            if (!video.isChecked() || age.isEmpty()) {
                Toast.makeText(this, "영상 동의와 연령대가 필요해요", Toast.LENGTH_SHORT).show();
                return;
            }
            Store.setConsent(this, true, bio.isChecked(), age);
            boolean wantBio = bio.isChecked();
            new Thread(() -> {   // 본인을 동의 인원 명부에 올리고 바로 얼굴 등록으로 (서버가 없으면 나중에 '개인정보 동의 인원'에서)
                try {
                    JSONObject r = Uploader.callJson(this, "GET", "/api/people", null);
                    boolean hasMe = false;
                    org.json.JSONArray ps = r.optJSONArray("people");
                    for (int i = 0; ps != null && i < ps.length(); i++) hasMe |= ps.optJSONObject(i).optString("relation").equals("본인");
                    if (!hasMe) {
                        JSONObject me = Uploader.callJson(this, "POST", "/api/people", new JSONObject().put("name", "나").put("relation", "본인")
                                .put("adult", true).put("age_band", age).put("agreed", true).put("bio", wantBio));
                        startActivity(new Intent(this, FaceActivity.class).putExtra("pid", me.optString("id")).putExtra("name", "나"));
                    }
                } catch (Exception ignored) {
                }
                runOnUiThread(this::finish);
            }).start();
        }));
    }
}
