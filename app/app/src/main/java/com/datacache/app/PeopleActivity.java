package com.datacache.app;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Intent;
import android.os.Bundle;
import android.util.Base64;
import android.view.Gravity;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONArray;
import org.json.JSONObject;

/**
 * 개인정보 동의 인원: 내 동의 · 내 얼굴 등록 · 함께 찍히는 가족(동의 받고 추가, 언제든 철회).
 * 동의한 가족은 녹화에서 회색으로 가리지 않고 얼굴만 가상 인물로 바꾼다. 동의하지 않은 사람·아이는 그대로 가린다.
 */
public class PeopleActivity extends Activity {
    private static final int PHOTO = 7, OWNER = 8;
    private static final String[] RELATIONS = {"가족", "배우자", "부모님", "형제자매", "친구", "같이 사는 사람"};
    private LinearLayout list;
    private TextView ownerDesc;
    private String relation = RELATIONS[0];
    private EditText name;
    private CheckBox video, sale;
    private TextView photoBtn;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        LinearLayout body = Ui.page(this, "개인정보 동의 인원", true);

        LinearLayout me = Ui.listCard(this);
        me.addView(Ui.row(this, Icon.PERSON, Ui.OK, Ui.OK_SOFT, "내 동의",
                Store.bioConsent(this) ? "영상·소리·움직임 + 워치 생체 동의함" : "영상·소리·움직임 동의함", null,
                v -> startActivity(new Intent(this, ConsentActivity.class))));
        LinearLayout face = Ui.row(this, Icon.FACE, Ui.PRIMARY, Ui.PRIMARY_SOFT, "내 얼굴 등록", "확인 중…", null,
                v -> startActivityForResult(new Intent(this, FaceActivity.class).putExtra("mode", FaceActivity.OWNER), OWNER));
        ownerDesc = (TextView) ((LinearLayout) face.getChildAt(1)).getChildAt(1);
        me.addView(face);
        Ui.add(this, body, me);

        body.addView(Ui.sectionTitle(this, "함께 찍히는 사람"));
        list = Ui.listCard(this);
        Ui.add(this, body, list);

        body.addView(Ui.sectionTitle(this, "동의 받고 추가하기"));
        LinearLayout add = Ui.card(this);
        name = Ui.input(this, "이름 (예: 엄마)", "", 0);
        add.addView(name, Ui.margins(this, -1, -2, 0, 0, 0, 0, 12));
        LinearLayout chips = new LinearLayout(this);
        chips.setOrientation(LinearLayout.VERTICAL);
        LinearLayout line = null;
        for (int i = 0; i < RELATIONS.length; i++) {
            if (i % 3 == 0) {
                line = new LinearLayout(this);
                chips.addView(line, Ui.margins(this, -1, -2, 0, 0, 0, 0, 8));
            }
            String r = RELATIONS[i];
            TextView c = Ui.chip(this, r, 0, 0);
            c.setTextSize(14);
            c.setGravity(Gravity.CENTER);
            c.setPadding(0, Ui.dp(this, 9), 0, Ui.dp(this, 9));
            c.setOnClickListener(v -> { relation = r; paint(chips); });
            line.addView(c, Ui.margins(this, 0, -2, 1f, 0, 0, i % 3 == 2 ? 0 : 8, 0));
        }
        paint(chips);
        add.addView(chips);
        video = check("(필수) 이 사람이 미션 영상에 나오는 것에 동의했어요");
        sale = check("(필수) 얼굴을 가상 인물로 바꾼 영상이 로봇 학습 데이터로 판매되는 것에 동의했어요");
        add.addView(video);
        add.addView(sale);
        TextView note = Ui.text(this, "아이(만 14세 미만)는 동의 인원으로 추가할 수 없고 항상 가려져요. 사진은 서버에 저장되지 않고 얼굴 특징값만 남아요.",
                12, Ui.SUB, false);
        note.setPadding(Ui.dp(this, 4), Ui.dp(this, 6), 0, Ui.dp(this, 12));
        add.addView(note);
        photoBtn = Ui.button(this, "얼굴 사진 찍기", false, v -> {
            FaceActivity.lastPhoto = null;
            startActivityForResult(new Intent(this, FaceActivity.class).putExtra("mode", FaceActivity.PERSON), PHOTO);
        });
        add.addView(photoBtn, Ui.margins(this, -1, -2, 0, 0, 0, 0, 10));
        add.addView(Ui.button(this, "동의 받고 추가", true, v -> submit()));
        Ui.add(this, body, add);
        refresh();
    }

    private CheckBox check(String s) {
        CheckBox c = new CheckBox(this);
        c.setText(s);
        c.setTextSize(14);
        c.setTextColor(Ui.INK);
        c.setButtonTintList(new android.content.res.ColorStateList(new int[][]{{android.R.attr.state_checked}, {}},
                new int[]{Ui.PRIMARY, Ui.LINE_DARK}));
        c.setPadding(Ui.dp(this, 6), Ui.dp(this, 8), 0, Ui.dp(this, 8));
        return c;
    }

    private void paint(LinearLayout chips) {
        for (int i = 0; i < chips.getChildCount(); i++) {
            LinearLayout l = (LinearLayout) chips.getChildAt(i);
            for (int j = 0; j < l.getChildCount(); j++) {
                TextView c = (TextView) l.getChildAt(j);
                boolean on = c.getText().toString().equals(relation);
                c.setTextColor(on ? android.graphics.Color.WHITE : Ui.SUB);
                c.setBackground(Ui.round(on ? Ui.PRIMARY : Ui.FIELD, Ui.dp(this, 99)));
            }
        }
    }

    private void refresh() {
        ownerDesc.setText(Store.ownerFace(this) ? "등록됨 · 녹화 속에서 나를 알아보는 데만 써요" : "아직 안 했어요 · 녹화 전 얼굴 확인에 필요해요");
        list.removeAllViews();
        list.addView(Ui.row(this, Icon.PERSON, Ui.SUB, Ui.FIELD, "불러오는 중…", null, null, null));
        new Thread(() -> {
            try {
                JSONObject r = Uploader.callJson(this, "GET", "/api/people", null);
                runOnUiThread(() -> show(r));
            } catch (Exception e) {
                runOnUiThread(() -> {
                    list.removeAllViews();
                    list.addView(Ui.row(this, Icon.CLOUD, Ui.SUB, Ui.FIELD, "서버에 연결 안 됨", "스마트 기기 연결에서 서버를 확인해 주세요", null, null));
                });
            }
        }).start();
    }

    private void show(JSONObject r) {
        Store.setOwnerFace(this, r.optBoolean("owner_registered"));
        ownerDesc.setText(Store.ownerFace(this) ? "등록됨 · 녹화 속에서 나를 알아보는 데만 써요" : "아직 안 했어요 · 녹화 전 얼굴 확인에 필요해요");
        list.removeAllViews();
        JSONArray ps = r.optJSONArray("people");
        if (ps == null || ps.length() == 0) {
            list.addView(Ui.row(this, Icon.PERSON, Ui.SUB, Ui.FIELD, "아직 없어요", "추가하지 않은 사람은 녹화에서 모두 가려져요", null, null));
            return;
        }
        for (int i = 0; i < ps.length(); i++) {
            JSONObject p = ps.optJSONObject(i);
            String id = p.optString("id"), nm = p.optString("name");
            TextView off = Ui.chip(this, "동의 철회", Ui.PRIMARY_DARK, Ui.PRIMARY_SOFT);
            off.setOnClickListener(v -> new AlertDialog.Builder(this)
                    .setMessage(nm + " 님의 동의를 철회할까요? 얼굴 특징값이 지워지고, 다음 녹화부터 다시 가려져요.")
                    .setPositiveButton("철회", (d, w) -> remove(id)).setNegativeButton("취소", null).show());
            list.addView(Ui.row(this, Icon.PERSON, Ui.OK, Ui.OK_SOFT, nm, p.optString("relation") + " · 영상·판매 동의 · 가상 얼굴로 바꿈", off, null));
        }
    }

    private void remove(String id) {
        new Thread(() -> {
            try {
                Uploader.callJson(this, "DELETE", "/api/people/" + id, null);
                runOnUiThread(this::refresh);
            } catch (Exception e) {
                runOnUiThread(() -> Toast.makeText(this, Uploader.reason(e), Toast.LENGTH_LONG).show());
            }
        }).start();
    }

    private void submit() {
        String nm = name.getText().toString().trim();
        if (nm.isEmpty()) { Toast.makeText(this, "이름을 적어 주세요", Toast.LENGTH_SHORT).show(); return; }
        if (!video.isChecked() || !sale.isChecked()) { Toast.makeText(this, "두 가지 모두 동의를 받아야 추가할 수 있어요", Toast.LENGTH_SHORT).show(); return; }
        byte[] photo = FaceActivity.lastPhoto;
        if (photo == null) { Toast.makeText(this, "얼굴 사진을 먼저 찍어 주세요", Toast.LENGTH_SHORT).show(); return; }
        new Thread(() -> {
            try {
                JSONObject b = new JSONObject().put("name", nm).put("relation", relation).put("consent_video", true).put("consent_sale", true)
                        .put("photo_b64", Base64.encodeToString(photo, Base64.NO_WRAP));
                Uploader.callJson(this, "POST", "/api/people", b);
                FaceActivity.lastPhoto = null;
                runOnUiThread(() -> {
                    name.setText("");
                    video.setChecked(false);
                    sale.setChecked(false);
                    photoBtn.setText("얼굴 사진 찍기");
                    Toast.makeText(this, nm + " 님을 추가했어요", Toast.LENGTH_SHORT).show();
                    refresh();
                });
            } catch (Exception e) {
                runOnUiThread(() -> Toast.makeText(this, Uploader.reason(e), Toast.LENGTH_LONG).show());
            }
        }).start();
    }

    @Override
    protected void onActivityResult(int req, int res, Intent data) {
        super.onActivityResult(req, res, data);
        if (req == PHOTO && res == RESULT_OK && FaceActivity.lastPhoto != null) photoBtn.setText("사진 찍음 ✓ · 다시 찍기");
        if (req == OWNER) refresh();
    }
}
