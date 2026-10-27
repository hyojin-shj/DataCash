package com.datacache.app;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Intent;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.view.Gravity;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.GridLayout;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;

/**
 * 개인정보 동의 인원: 함께 찍히는 사람(본인 포함)을 동의 받고 추가 → 얼굴 3장 등록 → (선택) 가상 인물 고르기.
 * 등록된 사람은 녹화에서 가리지 않고 얼굴만 가상 인물로 바꾼다. 등록 안 된 사람·아이는 가려진다.
 * 사람을 누르면: 얼굴 등록하기 · 가상 인물 고르기 · 생체 신호 동의(본인만) · 동의 철회.
 */
public class PeopleActivity extends Activity {
    private static final int FACE = 7;
    private static final String[] RELATIONS = {"본인", "배우자", "가족", "동거인", "기타"};
    private static final String[] AGES = {"19~29세", "30대", "40대", "50대", "60대 이상"};
    private LinearLayout list;
    private JSONArray donors = new JSONArray();
    private String relation = RELATIONS[0], age = "";
    private EditText name;
    private CheckBox adult, agree, bio;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        LinearLayout body = Ui.page(this, "개인정보 동의 인원", true);
        TextView intro = Ui.text(this, "영상에 함께 나오는 사람은 동의를 받고 얼굴을 등록해야 가려지지 않아요. 등록한 사람도 판매되는 영상에서는 실존하지 않는 가상 인물 얼굴로 바뀌어요.",
                13, Ui.SUB, false);
        intro.setPadding(Ui.dp(this, 4), 0, Ui.dp(this, 4), Ui.dp(this, 14));
        body.addView(intro);
        list = Ui.listCard(this);
        Ui.add(this, body, list);

        body.addView(Ui.sectionTitle(this, "동의 인원 추가"));
        LinearLayout add = Ui.card(this);
        name = Ui.input(this, "이름 또는 별칭", "", 0);
        add.addView(name, Ui.margins(this, -1, -2, 0, 0, 0, 0, 12));
        add.addView(Ui.text(this, "관계", 13, Ui.SUB, true));
        add.addView(chips(RELATIONS, relation, v -> { relation = v; bio.setEnabled(v.equals("본인")); if (!v.equals("본인")) bio.setChecked(false); }),
                Ui.margins(this, -1, -2, 0, 0, 8, 0, 8));
        adult = check("만 19세 이상 성인이에요");
        add.addView(adult);
        add.addView(Ui.text(this, "연령대", 13, Ui.SUB, true), Ui.margins(this, -2, -2, 0, 0, 8, 0, 0));
        add.addView(chips(AGES, "", v -> age = v), Ui.margins(this, -1, -2, 0, 0, 8, 0, 8));
        agree = check("(필수) 미션 영상에 나오는 것, 얼굴을 가상 인물로 바꾼 영상·소리·움직임이 로봇 학습 데이터로 판매되는 것에 동의해요. 언제든 철회할 수 있어요.");
        add.addView(agree);
        bio = check("(선택, 본인만) 워치의 심박수·심박 예비량(HRR)·피부온도 변화와 미션 후 응답을 함께 제공하는 데 동의해요 — 건강을 판단하거나 조언하지 않아요");
        add.addView(bio);
        TextView kid = Ui.text(this, "아이(만 19세 미만)는 추가할 수 없고 영상에서 항상 가려져요.", 12, Ui.PRIMARY_DARK, true);
        kid.setPadding(Ui.dp(this, 4), Ui.dp(this, 6), 0, Ui.dp(this, 12));
        add.addView(kid);
        add.addView(Ui.button(this, "동의하고 추가", true, v -> submit()));
        Ui.add(this, body, add);
    }

    @Override
    protected void onResume() {
        super.onResume();
        refresh();
    }

    interface Pick { void on(String v); }

    private LinearLayout chips(String[] opts, String sel, Pick on) {
        LinearLayout box = new LinearLayout(this);
        box.setOrientation(LinearLayout.VERTICAL);
        LinearLayout line = null;
        TextView[] all = new TextView[opts.length];
        for (int i = 0; i < opts.length; i++) {
            if (i % 3 == 0) { line = new LinearLayout(this); box.addView(line, Ui.margins(this, -1, -2, 0, 0, 0, 0, 8)); }
            String o = opts[i];
            TextView c = Ui.chip(this, o, 0, 0);
            c.setTextSize(14);
            c.setGravity(Gravity.CENTER);
            c.setPadding(0, Ui.dp(this, 9), 0, Ui.dp(this, 9));
            all[i] = c;
            c.setOnClickListener(v -> { on.on(o); for (TextView x : all) paintChip(x, x == c); });
            paintChip(c, o.equals(sel));
            line.addView(c, Ui.margins(this, 0, -2, 1f, 0, 0, 8, 0));
        }
        for (int k = opts.length % 3; k > 0 && k < 3; k++) line.addView(new android.view.View(this), Ui.margins(this, 0, 1, 1f, 0, 0, 8, 0));
        return box;
    }

    private void paintChip(TextView c, boolean on) {
        c.setTextColor(on ? android.graphics.Color.WHITE : Ui.SUB);
        c.setBackground(Ui.round(on ? Ui.PRIMARY : Ui.FIELD, Ui.dp(this, 99)));
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

    private void refresh() {
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
        donors = r.optJSONArray("donors") == null ? new JSONArray() : r.optJSONArray("donors");
        list.removeAllViews();
        JSONArray ps = r.optJSONArray("people");
        if (ps == null || ps.length() == 0) {
            list.addView(Ui.row(this, Icon.PERSON, Ui.SUB, Ui.FIELD, "아직 없어요", "먼저 '본인'을 추가하고 얼굴을 등록해 주세요 — 녹화 1단계에서 본인을 확인해요", null, null));
            return;
        }
        for (int i = 0; i < ps.length(); i++) {
            JSONObject p = ps.optJSONObject(i);
            String face = p.optString("face");
            StringBuilder d = new StringBuilder(p.optString("relation") + " · " + p.optString("age_band") + " · "
                    + p.optJSONObject("consent").optString("at").replace('T', ' ').substring(0, 10) + " 동의");
            JSONObject bc = p.optJSONObject("bio_consent");
            if (bc != null && bc.optBoolean("agreed")) d.append("\n심박수·HRR·피부온도 변화·미션 후 응답 제공 동의");
            if (face.equals("enrolled")) d.append("\n가상 인물: ").append(p.optString("donor").equals("auto") ? "자동 · 가장 덜 닮은 사람" : donorLabel(p.optString("donor")));
            if (face.equals("failed")) d.append("\n등록 실패 · ").append(p.optString("face_error"));
            if (!face.equals("enrolled")) d.append("\n눌러서 얼굴을 등록해야 영상에 남아요 (안 하면 가려져요)");
            TextView chip = face.equals("enrolled") ? Ui.chip(this, "얼굴 등록됨", Ui.OK, Ui.OK_SOFT)
                    : face.equals("failed") ? Ui.chip(this, "등록 실패", Ui.PRIMARY_DARK, Ui.PRIMARY_SOFT)
                    : Ui.chip(this, "얼굴 등록 필요", Ui.ACCENT_INK, Ui.ACCENT_SOFT);
            list.addView(Ui.row(this, Icon.PERSON, face.equals("enrolled") ? Ui.OK : Ui.SUB, face.equals("enrolled") ? Ui.OK_SOFT : Ui.FIELD,
                    p.optString("name"), d.toString(), chip, v -> menu(p)));
        }
    }

    private String donorLabel(String id) {
        for (int i = 0; i < donors.length(); i++) {
            JSONObject x = donors.optJSONObject(i);
            if (x.optString("id").equals(id)) return "가상 인물 " + (i + 1) + " (" + x.optString("label") + ")";
        }
        return id;
    }

    private void menu(JSONObject p) {
        String pid = p.optString("id"), nm = p.optString("name");
        boolean me = p.optString("relation").equals("본인");
        boolean bioOn = p.optJSONObject("bio_consent") != null && p.optJSONObject("bio_consent").optBoolean("agreed");
        java.util.List<String> items = new java.util.ArrayList<>();
        items.add(p.optString("face").equals("enrolled") ? "얼굴 다시 등록하기 (사진 3장)" : "얼굴 등록하기 (사진 3장)");
        items.add("가상 인물 고르기");
        if (me) items.add(bioOn ? "생체 신호 동의만 철회" : "생체 신호 제공 동의하기");
        items.add("동의 철회");
        new AlertDialog.Builder(this).setTitle(nm).setItems(items.toArray(new String[0]), (d, w) -> {
            String it = items.get(w);
            if (it.startsWith("얼굴")) enroll(pid, nm);
            else if (it.equals("가상 인물 고르기")) pickDonor(pid);
            else if (it.startsWith("생체")) post("/api/people/" + pid + "/consent", jsonBio(!bioOn),
                    bioOn ? "생체 신호 제공 동의를 철회했어요. 다음 녹화부터 심박·체온을 담지 않아요" : "생체 신호 제공에 동의했어요");
            else new AlertDialog.Builder(this).setMessage(nm + " 님의 동의를 철회할까요? 얼굴 특징값이 지워지고, 다음 녹화부터 가려져요.")
                        .setPositiveButton("철회", (a, b) -> delete(pid)).setNegativeButton("취소", null).show();
        }).show();
    }

    private static JSONObject jsonBio(boolean on) {
        try { return new JSONObject().put("bio", on); } catch (Exception e) { return new JSONObject(); }
    }

    private void enroll(String pid, String nm) {
        startActivityForResult(new Intent(this, FaceActivity.class).putExtra("pid", pid).putExtra("name", nm), FACE);
    }

    /** 가상 인물 고르기: 실존하지 않는 사람 사진 격자 + '자동 (가장 덜 닮은 사람)'. */
    private void pickDonor(String pid) {
        GridLayout g = new GridLayout(this);
        g.setColumnCount(3);
        g.setPadding(Ui.dp(this, 16), Ui.dp(this, 8), Ui.dp(this, 16), 0);
        AlertDialog[] dlg = new AlertDialog[1];
        int cell = (getResources().getDisplayMetrics().widthPixels - Ui.dp(this, 120)) / 3;
        for (int i = 0; i < donors.length(); i++) {
            JSONObject x = donors.optJSONObject(i);
            LinearLayout c = new LinearLayout(this);
            c.setOrientation(LinearLayout.VERTICAL);
            c.setGravity(Gravity.CENTER_HORIZONTAL);
            c.setPadding(Ui.dp(this, 4), Ui.dp(this, 4), Ui.dp(this, 4), Ui.dp(this, 8));
            ImageView iv = new ImageView(this);
            iv.setScaleType(ImageView.ScaleType.CENTER_CROP);
            iv.setBackground(Ui.round(Ui.FIELD, Ui.dp(this, 12)));
            iv.setClipToOutline(true);
            c.addView(iv, new LinearLayout.LayoutParams(cell, cell));
            c.addView(Ui.text(this, x.optString("label"), 12, Ui.SUB, false));
            String id = x.optString("id");
            c.setOnClickListener(v -> { post("/api/people/" + pid + "/donor", donorJson(id), "가상 인물을 바꿨어요"); dlg[0].dismiss(); });
            g.addView(c);
            new Thread(() -> {
                try {
                    HttpURLConnection h = (HttpURLConnection) new URL(Store.server(this) + "/api/donors/" + id + ".png").openConnection();
                    try (InputStream in = h.getInputStream()) {
                        android.graphics.Bitmap b = BitmapFactory.decodeStream(in);
                        runOnUiThread(() -> iv.setImageBitmap(b));
                    }
                } catch (Exception ignored) {
                }
            }).start();
        }
        dlg[0] = new AlertDialog.Builder(this).setTitle("가상 인물 고르기").setView(g)
                .setPositiveButton("자동 (가장 덜 닮은 사람)", (d, w) -> post("/api/people/" + pid + "/donor", donorJson("auto"), "가장 덜 닮은 사람으로 자동으로 골라요"))
                .setNegativeButton("취소", null).show();
    }

    private static JSONObject donorJson(String id) {
        try { return new JSONObject().put("donor", id); } catch (Exception e) { return new JSONObject(); }
    }

    private void post(String path, JSONObject body, String ok) {
        new Thread(() -> {
            try {
                Uploader.callJson(this, "POST", path, body);
                runOnUiThread(() -> { Toast.makeText(this, ok, Toast.LENGTH_SHORT).show(); refresh(); });
            } catch (Exception e) {
                runOnUiThread(() -> Toast.makeText(this, Uploader.reason(e), Toast.LENGTH_LONG).show());
            }
        }).start();
    }

    private void delete(String pid) {
        new Thread(() -> {
            try {
                Uploader.callJson(this, "DELETE", "/api/people/" + pid, null);
                runOnUiThread(this::refresh);
            } catch (Exception e) {
                runOnUiThread(() -> Toast.makeText(this, Uploader.reason(e), Toast.LENGTH_LONG).show());
            }
        }).start();
    }

    private void submit() {
        String nm = name.getText().toString().trim();
        if (nm.isEmpty()) { Toast.makeText(this, "이름이나 별칭을 적어 주세요", Toast.LENGTH_SHORT).show(); return; }
        if (!adult.isChecked()) { Toast.makeText(this, "만 19세 이상 성인만 추가할 수 있어요", Toast.LENGTH_SHORT).show(); return; }
        if (!agree.isChecked()) { Toast.makeText(this, "필수 동의를 받아야 추가할 수 있어요", Toast.LENGTH_SHORT).show(); return; }
        if (age.isEmpty()) { Toast.makeText(this, "연령대를 골라 주세요", Toast.LENGTH_SHORT).show(); return; }
        new Thread(() -> {
            try {
                JSONObject b = new JSONObject().put("name", nm).put("relation", relation).put("adult", true).put("age_band", age)
                        .put("agreed", true).put("bio", bio.isChecked() && relation.equals("본인"));
                JSONObject p = Uploader.callJson(this, "POST", "/api/people", b);
                if (relation.equals("본인")) Store.setConsent(this, true, bio.isChecked(), age);
                runOnUiThread(() -> {
                    name.setText("");
                    adult.setChecked(false);
                    agree.setChecked(false);
                    bio.setChecked(false);
                    Toast.makeText(this, nm + " 님을 추가했어요 · 얼굴을 등록해 주세요", Toast.LENGTH_SHORT).show();
                    enroll(p.optString("id"), nm);       // 추가하면 바로 얼굴 등록으로
                });
            } catch (Exception e) {
                runOnUiThread(() -> Toast.makeText(this, Uploader.reason(e), Toast.LENGTH_LONG).show());
            }
        }).start();
    }
}
