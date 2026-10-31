package com.datacash.app;

import android.app.Activity;
import android.app.AlertDialog;
import android.os.Bundle;
import android.view.Gravity;
import android.view.View;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.Locale;

/**
 * 이의 신청 현황: 신청마다 세 단계 (자동 확인 → 검토자 확인 → 결과)와 단계별 기록.
 * 인정된 추가 보상은 여기서 한 번만 적립한다. 신청은 내 녹화 기록에서 한다 (ask).
 */
public class AppealsActivity extends BaseActivity {
    /** 사유: 서버 값 · 보이는 이름. */
    static final String[] REASON_KEYS = {"reward", "failed", "privacy", "label", "other"};
    static final int[] REASON_NAMES = {R.string.reason_reward, R.string.reason_failed, R.string.reason_privacy, R.string.reason_label, R.string.reason_other};
    private LinearLayout list;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        LinearLayout body = Ui.page(this, getString(R.string.menu_appeals), true);
        TextView intro = Ui.text(this, getString(R.string.appeals_intro),
                13, Ui.SUB, false);
        intro.setPadding(Ui.dp(this, 4), 0, Ui.dp(this, 4), Ui.dp(this, 14));
        body.addView(intro);
        list = new LinearLayout(this);
        list.setOrientation(LinearLayout.VERTICAL);
        body.addView(list);
    }

    private final android.os.Handler ui = new android.os.Handler(android.os.Looper.getMainLooper());
    private boolean visible;

    @Override
    protected void onResume() {
        super.onResume();
        visible = true;
        tick();
    }

    @Override
    protected void onPause() {
        super.onPause();
        visible = false;
    }

    /** 떠 있는 동안 5초마다 (자동 확인·검토 결과가 바뀌면 보이게, 인정된 보상이 입금되면 '띠링'). */
    private void tick() {
        if (!visible) return;
        load();
        new Thread(() -> { Uploader.refreshPending(this); ui.post(() -> { if (visible) Payout.checkNew(this); }); }).start();
        ui.postDelayed(this::tick, 5000);
    }

    private void load() {
        new Thread(() -> {
            try {
                JSONArray as = Uploader.callJson(this, "GET", "/api/appeals", null).getJSONArray("appeals");
                runOnUiThread(() -> show(as));
            } catch (Exception e) {
                runOnUiThread(() -> {
                    list.removeAllViews();
                    LinearLayout c = Ui.listCard(this);
                    c.addView(Ui.row(this, Icon.CLOUD, Ui.SUB, Ui.FIELD, getString(R.string.server_offline), Uploader.reason(e), null, null));
                    Ui.add(this, list, c);
                });
            }
        }).start();
    }

    private void show(JSONArray as) {
        list.removeAllViews();
        int open = 0, ok = 0, no = 0, won = 0;
        java.util.Set<String> appealed = new java.util.HashSet<>();
        for (int i = 0; i < as.length(); i++) {
            JSONObject a = as.optJSONObject(i);
            String st = a.optString("state");
            appealed.add(a.optString("sid"));
            if (st.equals("accepted")) { ok++; won += a.optInt("extra_reward"); } else if (st.equals("rejected")) no++; else open++;
        }
        LinearLayout stats = Ui.card(this);
        stats.setOrientation(LinearLayout.HORIZONTAL);
        String[][] cells = {{getString(R.string.in_progress), String.valueOf(open)}, {getString(R.string.accepted), String.valueOf(ok)},
                {getString(R.string.dismissed), String.valueOf(no)}, {getString(R.string.recovered), Ui.won(this, won)}};
        for (String[] c : cells) {
            LinearLayout cell = new LinearLayout(this);
            cell.setOrientation(LinearLayout.VERTICAL);
            cell.setGravity(Gravity.CENTER_HORIZONTAL);
            cell.addView(Ui.text(this, c[0], 12, Ui.SUB, false));
            TextView v = Ui.text(this, c[1], 18, Ui.INK, true);
            v.setPadding(0, Ui.dp(this, 6), 0, 0);
            cell.addView(v);
            stats.addView(cell, Ui.weight());
        }
        Ui.add(this, list, stats);
        LinearLayout can = Ui.listCard(this);       // 반려됐는데 아직 신청 안 한 녹화
        for (java.io.File d : Store.sessions(this)) {
            JSONObject st = Uploader.localState(d);
            if (!st.optString("state").equals("rejected") || appealed.contains(d.getName())) continue;
            TextView b = Ui.chip(this, getString(R.string.appeal), Ui.PRIMARY_DARK, Ui.PRIMARY_SOFT);
            b.setOnClickListener(v -> ask(this, d.getName(), this::load));
            can.addView(Ui.row(this, Icon.FILM, Ui.SUB, Ui.FIELD, Store.titleOf(this, st, getString(R.string.recording)), getString(R.string.rejected_line, st.optString("error", "")), b, null));
        }
        java.util.List<JSONObject> going = new java.util.ArrayList<>(), done = new java.util.ArrayList<>();
        for (int i = 0; i < as.length(); i++) {
            JSONObject a = as.optJSONObject(i);
            (a.optString("state").equals("accepted") || a.optString("state").equals("rejected") ? done : going).add(a);
        }
        section(getString(R.string.in_progress), going, getString(R.string.no_in_progress));
        section(getString(R.string.results), done, getString(R.string.no_results));
        list.addView(Ui.sectionTitle(this, getString(R.string.appealable)));
        if (can.getChildCount() == 0) can.addView(Ui.row(this, Icon.LIST, Ui.SUB, Ui.FIELD, getString(R.string.no_rejected),
                getString(R.string.no_rejected_desc), null, null));
        Ui.add(this, list, can);
    }

    private void section(String title, java.util.List<JSONObject> items, String empty) {
        list.addView(Ui.sectionTitle(this, title));
        if (items.isEmpty()) {
            LinearLayout c = Ui.listCard(this);
            c.addView(Ui.row(this, Icon.LIST, Ui.SUB, Ui.FIELD, empty, null, null, null));
            Ui.add(this, list, c);
            return;
        }
        for (JSONObject a : items) {
            String state = a.optString("state");
            int extra = a.optInt("extra_reward");
            LinearLayout c = Ui.card(this);
            LinearLayout top = new LinearLayout(this);
            top.setGravity(Gravity.CENTER_VERTICAL);
            top.addView(Ui.text(this, a.optString("reason_ko"), 17, Ui.INK, true), Ui.weight());
            int fg = state.equals("accepted") ? Ui.OK : state.equals("rejected") ? Ui.PRIMARY_DARK : Ui.ACCENT_INK;
            int bg = state.equals("accepted") ? Ui.OK_SOFT : state.equals("rejected") ? Ui.PRIMARY_SOFT : Ui.ACCENT_SOFT;
            String label = state.equals("accepted") ? (extra > 0 ? getString(R.string.accepted_plus, extra) : getString(R.string.accepted))
                    : getString(state.equals("rejected") ? R.string.not_accepted : state.equals("reprocessing") ? R.string.reprocessing : R.string.reviewing);
            top.addView(Ui.chip(this, label, fg, bg));
            c.addView(top);
            TextView sid = Ui.text(this, getString(R.string.recording_sid, a.optString("sid")) + (a.optString("detail").isEmpty() ? "" : " · " + a.optString("detail")),
                    12, Ui.SUB, false);
            sid.setPadding(0, Ui.dp(this, 6), 0, Ui.dp(this, 12));
            c.addView(sid);
            c.addView(stages(state));
            JSONArray steps = a.optJSONArray("steps");
            for (int k = 0; steps != null && k < steps.length(); k++) {
                JSONObject s = steps.optJSONObject(k);
                if (s.optString("stage").equals("filed")) continue;
                TextView t = Ui.text(this, "• " + s.optString("note"), 13, Ui.INK, false);
                t.setPadding(0, Ui.dp(this, 8), 0, 0);
                c.addView(t);
            }
            Ui.add(this, list, c);
        }
    }

    /** 세 단계 진행 표시: 자동 확인 · 검토자 확인 · 결과. */
    private View stages(String state) {
        int reached = state.equals("accepted") || state.equals("rejected") ? 3 : state.equals("review") ? 2 : 1;
        LinearLayout row = new LinearLayout(this);
        String[] names = {getString(R.string.stage_auto), getString(R.string.stage_reviewer), getString(R.string.stage_result)};
        for (int i = 0; i < 3; i++) {
            boolean on = i < reached;
            TextView t = Ui.text(this, names[i], 12, on ? Ui.PRIMARY_DARK : Ui.LINE_DARK, true);
            t.setGravity(Gravity.CENTER);
            t.setPadding(0, Ui.dp(this, 8), 0, Ui.dp(this, 8));
            t.setBackground(Ui.round(on ? Ui.PRIMARY_SOFT : Ui.FIELD, Ui.dp(this, 10)));
            row.addView(t, Ui.margins(this, 0, -2, 1f, 0, 0, i < 2 ? 6 : 0, 0));
        }
        return row;
    }

    /** 녹화 하나에 이의 신청 (사유 고르기 + 내용). 끝나면 done 을 부른다. */
    static void ask(Activity a, String sid, Runnable done) {
        LinearLayout box = new LinearLayout(a);
        box.setOrientation(LinearLayout.VERTICAL);
        box.setPadding(Ui.dp(a, 20), Ui.dp(a, 8), Ui.dp(a, 20), 0);
        final String[] pick = {REASON_KEYS[0]};
        LinearLayout chips = new LinearLayout(a);
        chips.setOrientation(LinearLayout.VERTICAL);
        for (int k = 0; k < REASON_KEYS.length; k++) {
            String key = REASON_KEYS[k];
            TextView c = Ui.text(a, a.getString(REASON_NAMES[k]), 15, Ui.INK, false);
            c.setTag(key);
            c.setPadding(Ui.dp(a, 14), Ui.dp(a, 11), Ui.dp(a, 14), Ui.dp(a, 11));
            c.setOnClickListener(v -> {
                pick[0] = key;
                for (int i = 0; i < chips.getChildCount(); i++) {
                    TextView x = (TextView) chips.getChildAt(i);
                    boolean on = x.getTag().equals(pick[0]);
                    x.setBackground(Ui.round(on ? Ui.PRIMARY_SOFT : 0, Ui.dp(a, 12)));
                    x.setTextColor(on ? Ui.PRIMARY_DARK : Ui.INK);
                }
            });
            chips.addView(c, Ui.margins(a, -1, -2, 0, 0, 0, 0, 4));
        }
        chips.getChildAt(0).performClick();
        box.addView(chips);
        EditText detail = Ui.input(a, a.getString(R.string.detail_hint), "", 0);
        box.addView(detail, Ui.margins(a, -1, -2, 0, 0, 8, 0, 0));
        new AlertDialog.Builder(a).setTitle(R.string.appeal).setView(box).setNegativeButton(R.string.cancel, null)
                .setPositiveButton(R.string.submit, (d, w) -> new Thread(() -> {
                    try {
                        JSONObject r = Uploader.callJson(a, "POST", "/api/sessions/" + sid + "/appeal",
                                new JSONObject().put("reason", pick[0]).put("detail", detail.getText().toString()));
                        String last = r.getJSONArray("steps").getJSONObject(r.getJSONArray("steps").length() - 1).optString("note");
                        a.runOnUiThread(() -> { Toast.makeText(a, last, Toast.LENGTH_LONG).show(); done.run(); });
                    } catch (Exception e) {
                        a.runOnUiThread(() -> Toast.makeText(a, Uploader.reason(e), Toast.LENGTH_LONG).show());
                    }
                }).start()).show();
    }
}
