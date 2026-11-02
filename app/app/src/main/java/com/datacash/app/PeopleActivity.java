package com.datacash.app;

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
public class PeopleActivity extends BaseActivity {
    private static final int FACE = 7;
    private String[] RELATIONS, AGES, relNames, ageNames;    // 서버 값 · 보이는 이름 (같은 순서)
    private String self;                                     // 서버 관계 값 '본인'
    private LinearLayout list;
    private JSONArray donors = new JSONArray();
    private String relation, age = "";
    private EditText name;
    private CheckBox adult, agree, bio;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        RELATIONS = getResources().getStringArray(R.array.relation_values);
        AGES = getResources().getStringArray(R.array.age_band_values);
        relNames = getResources().getStringArray(R.array.relations);
        ageNames = getResources().getStringArray(R.array.age_bands);
        self = RELATIONS[0];
        relation = self;
        LinearLayout body = Ui.page(this, getString(R.string.menu_people), true);
        TextView intro = Ui.text(this, getString(R.string.people_intro),
                13, Ui.SUB, false);
        intro.setPadding(Ui.dp(this, 4), 0, Ui.dp(this, 4), Ui.dp(this, 14));
        body.addView(intro);
        list = Ui.listCard(this);
        Ui.add(this, body, list);

        body.addView(Ui.sectionTitle(this, getString(R.string.people_add_title)));
        LinearLayout add = Ui.card(this);
        name = Ui.input(this, getString(R.string.name_hint), "", 0);
        add.addView(name, Ui.margins(this, -1, -2, 0, 0, 0, 0, 12));
        add.addView(Ui.text(this, getString(R.string.relation), 13, Ui.SUB, true));
        add.addView(chips(RELATIONS, relNames, relation, v -> { relation = v; bio.setEnabled(v.equals(self)); if (!v.equals(self)) bio.setChecked(false); }),
                Ui.margins(this, -1, -2, 0, 0, 8, 0, 8));
        adult = check(getString(R.string.adult_check));
        add.addView(adult);
        add.addView(Ui.text(this, getString(R.string.age_band), 13, Ui.SUB, true), Ui.margins(this, -2, -2, 0, 0, 8, 0, 0));
        add.addView(chips(AGES, ageNames, "", v -> age = v), Ui.margins(this, -1, -2, 0, 0, 8, 0, 8));
        agree = check(getString(R.string.people_agree));
        add.addView(agree);
        bio = check(getString(R.string.people_bio));
        add.addView(bio);
        TextView kid = Ui.text(this, getString(R.string.kid_note), 12, Ui.PRIMARY_DARK, true);
        kid.setPadding(Ui.dp(this, 4), Ui.dp(this, 6), 0, Ui.dp(this, 12));
        add.addView(kid);
        add.addView(Ui.button(this, getString(R.string.agree_add), true, v -> submit()));
        Ui.add(this, body, add);
    }

    @Override
    protected void onResume() {
        super.onResume();
        refresh();
    }

    interface Pick { void on(String v); }

    /** 보이는 이름: 서버 값(한국어) → 지금 언어. 모르는 값은 그대로. */
    private static String label(String[] vals, String[] names, String v) {
        for (int i = 0; i < vals.length && i < names.length; i++) if (vals[i].equals(v)) return names[i];
        return v;
    }

    private LinearLayout chips(String[] opts, String[] names, String sel, Pick on) {
        LinearLayout box = new LinearLayout(this);
        box.setOrientation(LinearLayout.VERTICAL);
        LinearLayout line = null;
        TextView[] all = new TextView[opts.length];
        for (int i = 0; i < opts.length; i++) {
            if (i % 3 == 0) { line = new LinearLayout(this); box.addView(line, Ui.margins(this, -1, -2, 0, 0, 0, 0, 8)); }
            String o = opts[i];
            TextView c = Ui.chip(this, i < names.length ? names[i] : o, 0, 0);
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
                    list.addView(Ui.row(this, Icon.CLOUD, Ui.SUB, Ui.FIELD, getString(R.string.server_offline), getString(R.string.check_server_in_devices), null, null));
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
            list.addView(Ui.row(this, Icon.PERSON, Ui.SUB, Ui.FIELD, getString(R.string.none_yet), getString(R.string.none_yet_desc), null, null));
            return;
        }
        for (int i = 0; i < ps.length(); i++) {
            JSONObject p = ps.optJSONObject(i);
            String face = p.optString("face");
            StringBuilder d = new StringBuilder(label(RELATIONS, relNames, p.optString("relation")) + " · " + label(AGES, ageNames, p.optString("age_band")) + " · "
                    + getString(R.string.consented_on, p.optJSONObject("consent").optString("at").replace('T', ' ').substring(0, 10)));
            JSONObject bc = p.optJSONObject("bio_consent");
            if (bc != null && bc.optBoolean("agreed")) d.append("\n").append(getString(R.string.bio_consented_line));
            if (face.equals("enrolled")) d.append("\n").append(getString(R.string.virtual_person_line,
                    p.optString("donor").equals("auto") ? getString(R.string.donor_auto_desc) : donorLabel(p.optString("donor"))));
            if (face.equals("failed")) d.append("\n").append(getString(R.string.enroll_failed_line, getString(FaceActivity.errorText(p.optString("face_error_code")))));
            if (!face.equals("enrolled")) d.append("\n").append(getString(R.string.enroll_needed_line));
            TextView chip = face.equals("enrolled") ? Ui.chip(this, getString(R.string.face_enrolled), Ui.OK, Ui.OK_SOFT)
                    : face.equals("failed") ? Ui.chip(this, getString(R.string.enroll_failed), Ui.PRIMARY_DARK, Ui.PRIMARY_SOFT)
                    : Ui.chip(this, getString(R.string.face_needed), Ui.ACCENT_INK, Ui.ACCENT_SOFT);
            list.addView(Ui.row(this, Icon.PERSON, face.equals("enrolled") ? Ui.OK : Ui.SUB, face.equals("enrolled") ? Ui.OK_SOFT : Ui.FIELD,
                    p.optString("name"), d.toString(), chip, v -> menu(p)));
        }
    }

    private String donorLabel(String id) {
        for (int i = 0; i < donors.length(); i++) {
            JSONObject x = donors.optJSONObject(i);
            if (x.optString("id").equals(id)) return getString(R.string.donor_n, i + 1, x.optString("label"));
        }
        return id;
    }

    private void menu(JSONObject p) {
        String pid = p.optString("id"), nm = p.optString("name");
        boolean me = p.optString("relation").equals(self);
        boolean bioOn = p.optJSONObject("bio_consent") != null && p.optJSONObject("bio_consent").optBoolean("agreed");
        java.util.List<String> items = new java.util.ArrayList<>();
        java.util.List<String> acts = new java.util.ArrayList<>();     // 고른 줄 → 할 일
        items.add(getString(p.optString("face").equals("enrolled") ? R.string.re_enroll : R.string.enroll)); acts.add("face");
        items.add(getString(R.string.pick_donor)); acts.add("donor");
        if (me) { items.add(getString(bioOn ? R.string.bio_withdraw : R.string.bio_agree)); acts.add("bio"); }
        items.add(getString(R.string.withdraw_consent)); acts.add("withdraw");
        new AlertDialog.Builder(this).setTitle(nm).setItems(items.toArray(new String[0]), (d, w) -> {
            String it = acts.get(w);
            if (it.equals("face")) enroll(pid, nm);
            else if (it.equals("donor")) pickDonor(pid);
            else if (it.equals("bio")) post("/api/people/" + pid + "/consent", jsonBio(!bioOn),
                    getString(bioOn ? R.string.bio_withdrawn : R.string.bio_agreed));
            else new AlertDialog.Builder(this).setMessage(getString(R.string.withdraw_confirm, nm))
                        .setPositiveButton(R.string.withdraw, (a, b) -> delete(pid)).setNegativeButton(R.string.cancel, null).show();
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
            c.setOnClickListener(v -> { post("/api/people/" + pid + "/donor", donorJson(id), getString(R.string.donor_changed)); dlg[0].dismiss(); });
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
        dlg[0] = new AlertDialog.Builder(this).setTitle(R.string.pick_donor).setView(g)
                .setPositiveButton(R.string.donor_auto_btn, (d, w) -> post("/api/people/" + pid + "/donor", donorJson("auto"), getString(R.string.donor_auto_toast)))
                .setNegativeButton(R.string.cancel, null).show();
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
        if (nm.isEmpty()) { Toast.makeText(this, R.string.need_name, Toast.LENGTH_SHORT).show(); return; }
        if (!adult.isChecked()) { Toast.makeText(this, R.string.adult_only, Toast.LENGTH_SHORT).show(); return; }
        if (!agree.isChecked()) { Toast.makeText(this, R.string.need_agree, Toast.LENGTH_SHORT).show(); return; }
        if (age.isEmpty()) { Toast.makeText(this, R.string.pick_age, Toast.LENGTH_SHORT).show(); return; }
        new Thread(() -> {
            try {
                JSONObject b = new JSONObject().put("name", nm).put("relation", relation).put("adult", true).put("age_band", age)
                        .put("agreed", true).put("bio", bio.isChecked() && relation.equals(self));
                JSONObject p = Uploader.callJson(this, "POST", "/api/people", b);
                if (relation.equals(self)) Store.setConsent(this, true, bio.isChecked(), age);
                runOnUiThread(() -> {
                    name.setText("");
                    adult.setChecked(false);
                    agree.setChecked(false);
                    bio.setChecked(false);
                    Toast.makeText(this, getString(R.string.added_enroll, nm), Toast.LENGTH_SHORT).show();
                    enroll(p.optString("id"), nm);       // 추가하면 바로 얼굴 등록으로
                });
            } catch (Exception e) {
                runOnUiThread(() -> Toast.makeText(this, Uploader.reason(e), Toast.LENGTH_LONG).show());
            }
        }).start();
    }
}
