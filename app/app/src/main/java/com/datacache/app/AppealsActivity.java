package com.datacache.app;

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
public class AppealsActivity extends Activity {
    static final String[][] REASONS = {{"reward", "보상이 적어요"}, {"failed", "처리가 실패했어요"}, {"privacy", "가려야 할 게 보여요"},
            {"label", "라벨·단계가 틀려요"}, {"other", "기타"}};
    private LinearLayout list;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        LinearLayout body = Ui.page(this, "이의 신청 현황", true);
        TextView intro = Ui.text(this, "자동 확인으로 바로 끝나지 않으면 검토자가 확인해요. '가려야 할 게 보여요'는 신청하는 즉시 판매에서 빠져요.",
                13, Ui.SUB, false);
        intro.setPadding(Ui.dp(this, 4), 0, Ui.dp(this, 4), Ui.dp(this, 14));
        body.addView(intro);
        list = new LinearLayout(this);
        list.setOrientation(LinearLayout.VERTICAL);
        body.addView(list);
        load();
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
                    c.addView(Ui.row(this, Icon.CLOUD, Ui.SUB, Ui.FIELD, "서버에 연결 안 됨", Uploader.reason(e), null, null));
                    Ui.add(this, list, c);
                });
            }
        }).start();
    }

    private void show(JSONArray as) {
        list.removeAllViews();
        if (as.length() == 0) {
            LinearLayout c = Ui.listCard(this);
            c.addView(Ui.row(this, Icon.LIST, Ui.SUB, Ui.FIELD, "이의 신청이 없어요", "내 녹화 기록에서 녹화마다 신청할 수 있어요", null, null));
            Ui.add(this, list, c);
            return;
        }
        for (int i = 0; i < as.length(); i++) {
            JSONObject a = as.optJSONObject(i);
            String state = a.optString("state");
            int extra = a.optInt("extra_reward");
            if (state.equals("accepted") && extra > 0) Store.creditOnce(this, "appeal:" + a.optString("id"), extra);
            LinearLayout c = Ui.card(this);
            LinearLayout top = new LinearLayout(this);
            top.setGravity(Gravity.CENTER_VERTICAL);
            top.addView(Ui.text(this, a.optString("reason_ko"), 17, Ui.INK, true), Ui.weight());
            int fg = state.equals("accepted") ? Ui.OK : state.equals("rejected") ? Ui.PRIMARY_DARK : Ui.ACCENT_INK;
            int bg = state.equals("accepted") ? Ui.OK_SOFT : state.equals("rejected") ? Ui.PRIMARY_SOFT : Ui.ACCENT_SOFT;
            String label = state.equals("accepted") ? (extra > 0 ? String.format(Locale.KOREA, "인정 +%,d원", extra) : "인정")
                    : state.equals("rejected") ? "인정 안 됨" : state.equals("reprocessing") ? "다시 처리 중" : "검토 중";
            top.addView(Ui.chip(this, label, fg, bg));
            c.addView(top);
            TextView sid = Ui.text(this, "녹화 " + a.optString("sid") + (a.optString("detail").isEmpty() ? "" : " · " + a.optString("detail")),
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
        String[] names = {"자동 확인", "검토자 확인", "결과"};
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
        final String[] pick = {REASONS[0][0]};
        LinearLayout chips = new LinearLayout(a);
        chips.setOrientation(LinearLayout.VERTICAL);
        for (String[] r : REASONS) {
            TextView c = Ui.text(a, r[1], 15, Ui.INK, false);
            c.setTag(r[0]);
            c.setPadding(Ui.dp(a, 14), Ui.dp(a, 11), Ui.dp(a, 14), Ui.dp(a, 11));
            c.setOnClickListener(v -> {
                pick[0] = r[0];
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
        EditText detail = Ui.input(a, "자세히 (선택)", "", 0);
        box.addView(detail, Ui.margins(a, -1, -2, 0, 0, 8, 0, 0));
        new AlertDialog.Builder(a).setTitle("이의 신청").setView(box).setNegativeButton("취소", null)
                .setPositiveButton("신청", (d, w) -> new Thread(() -> {
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
