package com.datacash.app;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.view.Gravity;
import android.widget.LinearLayout;
import android.widget.TextView;

import org.json.JSONObject;

import java.io.File;
import java.util.Locale;

/** 내 녹화 기록: 확정 보상·검수 중 요약 + 녹화마다 상태 뱃지, 올리기 전이면 다시 올리기, 결과 보기. */
public class HistoryActivity extends BaseActivity {
    private final android.os.Handler ui = new android.os.Handler(android.os.Looper.getMainLooper());
    private boolean visible;

    /** 떠 있는 동안 5초마다 처리 중인 녹화의 결과를 받아 다시 그리고, 입금되면 '띠링'. */
    private void poll() {
        if (!visible) return;
        new Thread(() -> {
            String before = Uploader.signature(this);
            Uploader.refreshPending(this);
            boolean changed = !before.equals(Uploader.signature(this));
            ui.post(() -> { if (visible) { if (changed) render(); Payout.checkNew(this); ui.postDelayed(this::poll, 5000); } });
        }).start();
    }

    @Override
    protected void onPause() {
        super.onPause();
        visible = false;
    }

    @Override
    protected void onResume() {
        super.onResume();
        visible = true;
        render();
        ui.postDelayed(this::poll, 1500);
    }

    private void render() {
        LinearLayout body = Ui.page(this, getString(R.string.menu_history), true);
        File[] ss = Store.sessions(this);
        int confirmed = 0, reviewing = 0;
        for (File d : ss) {
            JSONObject st = Uploader.localState(d);
            String s = st.optString("state", "saved");
            if (s.equals("done")) confirmed += st.optInt("reward");
            else if (s.equals("queued") || s.equals("processing") || s.equals("uploading")) reviewing++;
        }
        LinearLayout sum = Ui.card(this);
        sum.setOrientation(LinearLayout.HORIZONTAL);
        sum.addView(stat(getString(R.string.history_confirmed), Ui.won(this, confirmed), Ui.INK), Ui.weight());
        sum.addView(stat(getString(R.string.history_reviewing), getString(R.string.count_items, reviewing), reviewing > 0 ? Ui.PRIMARY : Ui.LINE_DARK), Ui.weight());
        Ui.add(this, body, sum);

        LinearLayout list = Ui.listCard(this);
        if (ss.length == 0) {
            list.addView(Ui.row(this, Icon.FILM, Ui.SUB, Ui.FIELD, getString(R.string.no_recordings), getString(R.string.no_recordings_desc), null, null));
        }
        for (File d : ss) {
            if (!new File(d, "meta.json").exists()) continue;   // 녹화 중 끊긴 폴더
            list.addView(item(d));
        }
        Ui.add(this, body, list);
    }

    private LinearLayout stat(String label, String value, int color) {
        LinearLayout c = new LinearLayout(this);
        c.setOrientation(LinearLayout.VERTICAL);
        c.addView(Ui.text(this, label, 13, Ui.SUB, false));
        TextView v = Ui.text(this, value, 24, color, true);
        v.setPadding(0, Ui.dp(this, 8), 0, 0);
        c.addView(v);
        return c;
    }

    private LinearLayout item(File d) {
        JSONObject st = Uploader.localState(d);
        String s = st.optString("state", "saved");
        String n = d.getName();   // yyyyMMdd_HHmmss…
        String when = n.length() >= 13 ? String.format("%s.%s.%s %s:%s", n.substring(0, 4), n.substring(4, 6), n.substring(6, 8),
                n.substring(9, 11), n.substring(11, 13)) : n;
        String sub = when;
        if (s.equals("done")) {
            sub += " · " + getString(R.string.minutes_level, st.optDouble("minutes"), levelLabel(this, st.optString("level")));
            JSONObject py = st.optJSONObject("payout");
            if (py != null) sub += py.optString("status").equals("sent") ? "\n" + getString(R.string.paid_done, py.optString("label"))
                    : "\n" + getString(R.string.link_to_get_paid);
            if (st.optInt("appeal_extra") > 0) sub += "\n" + getString(R.string.appeal_extra, st.optInt("appeal_extra"));
        } else if (s.equals("collected")) {
            sub += "\n" + getString(R.string.collected_later);
        } else if (s.equals("rejected")) {
            sub += "\n" + getString(R.string.reason_line, st.optString("error", getString(R.string.reason_default)));
        }
        int fg = s.equals("done") ? Ui.OK : s.equals("rejected") || s.equals("failed") || s.equals("upload_failed") ? Ui.PRIMARY_DARK : Ui.SUB;
        int bg = s.equals("done") ? Ui.OK_SOFT : fg == Ui.PRIMARY_DARK ? Ui.PRIMARY_SOFT : Ui.FIELD;
        String chip = s.equals("done") && st.optInt("reward") > 0 ? Ui.plusWon(this, st.optInt("reward") + st.optInt("appeal_extra"))
                : s.equals("saved") ? getString(R.string.before_upload) : Uploader.stateLabel(this, s);
        LinearLayout box = new LinearLayout(this);
        box.setOrientation(LinearLayout.VERTICAL);
        box.addView(Ui.row(this, Icon.FILM, Ui.SUB, Ui.FIELD, st.optString("title", getString(R.string.recording)), sub, Ui.chip(this, chip, fg, bg), null));
        LinearLayout acts = new LinearLayout(this);
        acts.setOrientation(LinearLayout.VERTICAL);
        acts.setPadding(Ui.dp(this, 70), 0, Ui.dp(this, 8), Ui.dp(this, 12));
        if (s.equals("saved") || s.equals("upload_failed")) {
            TextView up = Ui.button(this, getString(R.string.upload_again), false, v -> {
                Uploader.save(d, "state", "saved");
                ProcessService.enqueue(this, d);
                android.widget.Toast.makeText(this, R.string.uploading_toast, android.widget.Toast.LENGTH_SHORT).show();
                recreate();
            });
            up.setMinHeight(Ui.dp(this, 44));
            acts.addView(up, Ui.margins(this, -1, -2, 0, 0, 0, 0, 8));
        }
        LinearLayout links = new LinearLayout(this);
        TextView more = Ui.text(this, getString(R.string.view_result), 13, Ui.PRIMARY, true);
        more.setPadding(0, Ui.dp(this, 2), Ui.dp(this, 18), 0);
        more.setOnClickListener(v -> open(d, false));
        links.addView(more);
        if (s.equals("done") || s.equals("rejected")) {
            TextView pv = Ui.text(this, getString(R.string.preview_link), 13, Ui.PRIMARY, true);
            pv.setPadding(0, Ui.dp(this, 2), Ui.dp(this, 18), 0);
            pv.setOnClickListener(v -> startActivity(new Intent(this, PreviewActivity.class).putExtra("dir", d.getAbsolutePath())));
            links.addView(pv);
        }
        if (!s.equals("saved") && !s.equals("uploading")) {     // 서버에 올라간 녹화만
            TextView ap = Ui.text(this, getString(R.string.appeal_link), 13, Ui.SUB, true);
            ap.setPadding(0, Ui.dp(this, 2), 0, 0);
            ap.setOnClickListener(v -> AppealsActivity.ask(this, d.getName(), () -> startActivity(new Intent(this, AppealsActivity.class))));
            links.addView(ap);
        }
        acts.addView(links, Ui.wrap());
        box.addView(acts);
        box.setGravity(Gravity.START);
        return box;
    }

    private void open(File d, boolean upload) {
        startActivity(new Intent(this, ResultActivity.class).putExtra("dir", d.getAbsolutePath()).putExtra("upload", upload));
    }

    static String levelLabel(android.content.Context c, String level) {
        switch (level) {
            case "bio": return c.getString(R.string.level_bio);
            case "premium": return c.getString(R.string.level_premium);
            case "basic": return c.getString(R.string.level_basic);
            default: return level;
        }
    }
}
