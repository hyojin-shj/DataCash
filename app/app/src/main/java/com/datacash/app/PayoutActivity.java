package com.datacash.app;

import android.app.Activity;
import android.os.Bundle;
import android.text.InputType;
import android.widget.EditText;
import android.widget.HorizontalScrollView;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONObject;

import java.io.File;
import java.util.Locale;

/**
 * 보상 받을 계좌: 받은 보상 · 계좌 연결을 기다리는 보상 · 은행 고르기 + 계좌번호 · 입금 내역.
 * 계좌번호 전체는 이 폰에만 저장하고 서버에는 '은행 ****1234' 표시만 보낸다. 시험 버전이라 실제 송금은 하지 않는다.
 * 계좌를 연결하면 기다리던 보상이 바로 '입금'되고 '띠링'.
 */
public class PayoutActivity extends Activity {
    private static final String[] BANKS = {"KB국민", "신한", "우리", "하나", "NH농협", "IBK기업", "카카오뱅크", "토스뱅크", "케이뱅크"};
    private String bank;
    private boolean editing;
    private TextView total, status;

    @Override
    protected void onResume() {
        super.onResume();
        render();
        new Thread(() -> {
            try {
                JSONObject r = Uploader.callJson(this, "GET", "/api/payout/account", null);
                runOnUiThread(() -> showServer(r));
            } catch (Exception ignored) {
            }
        }).start();
    }

    private void showServer(JSONObject r) {
        total.setText(String.format(Locale.KOREA, "%,d원", r.optInt("total")));
        int waiting = r.optInt("waiting");
        if (waiting > 0) status.setText(String.format(Locale.KOREA, "입금 기다리는 보상 %,d원 · 계좌를 연결하면 바로 들어와요", waiting));
    }

    private void render() {
        LinearLayout body = Ui.page(this, "보상 받을 계좌", true);
        bank = Store.bank(this).isEmpty() ? BANKS[0] : Store.bank(this);
        LinearLayout top = Ui.card(this);
        top.addView(Ui.text(this, "받은 보상", 13, Ui.SUB, false));
        total = Ui.text(this, "-", 28, Ui.INK, true);
        total.setPadding(0, Ui.dp(this, 8), 0, Ui.dp(this, 8));
        top.addView(total);
        status = Ui.text(this, Store.account(this).isEmpty() ? "아직 계좌를 연결하지 않았어요" : label() + " 으로 받아요", 13, Ui.SUB, false);
        top.addView(status);
        Ui.add(this, body, top);

        body.addView(Ui.sectionTitle(this, "입금받을 계좌"));
        if (!Store.account(this).isEmpty() && !editing) {
            LinearLayout c = Ui.listCard(this);
            TextView change = Ui.chip(this, "계좌 바꾸기", Ui.SUB, Ui.FIELD);
            change.setOnClickListener(v -> { editing = true; render(); });
            c.addView(Ui.row(this, Icon.STAR, Ui.ACCENT_INK, Ui.ACCENT_SOFT, Store.bank(this), Store.maskedAccount(this) + " · 연결됨", change, null));
            Ui.add(this, body, c);
        } else {
            LinearLayout acc = Ui.card(this);
            HorizontalScrollView hs = new HorizontalScrollView(this);
            hs.setHorizontalScrollBarEnabled(false);
            LinearLayout chips = new LinearLayout(this);
            hs.addView(chips);
            for (String b : BANKS) {
                TextView c = Ui.chip(this, b, 0, 0);
                c.setTextSize(14);
                c.setPadding(Ui.dp(this, 14), Ui.dp(this, 9), Ui.dp(this, 14), Ui.dp(this, 9));
                c.setOnClickListener(v -> { bank = b; paint(chips); });
                chips.addView(c, Ui.margins(this, -2, -2, 0, 0, 0, 8, 0));
            }
            paint(chips);
            acc.addView(hs);
            EditText num = Ui.input(this, "계좌번호 (- 없이)", "", InputType.TYPE_CLASS_NUMBER);
            acc.addView(num, Ui.margins(this, -1, -2, 0, 0, 14, 0, 14));
            acc.addView(Ui.button(this, "계좌 연결하고 바로 받기", true, v -> save(num.getText().toString().replaceAll("[^0-9]", ""))));
            Ui.add(this, body, acc);
        }
        TextView note = Ui.text(this, "계좌번호 전체는 이 폰에만 저장되고, 서버에는 가린 번호만 보내요. 시험 버전이라 실제 송금 대신 시험 지급으로 기록돼요.", 12, Ui.SUB, false);
        note.setPadding(Ui.dp(this, 4), 0, Ui.dp(this, 4), Ui.dp(this, 6));
        body.addView(note);

        body.addView(Ui.sectionTitle(this, "입금 내역"));
        LinearLayout list = Ui.listCard(this);
        int n = 0;
        for (File d : Store.sessions(this)) {
            JSONObject st = Uploader.localState(d);
            JSONObject py = st.optJSONObject("payout");
            if (py == null) continue;
            String id = d.getName();
            boolean sent = py.optString("status").equals("sent");
            TextView right = sent ? Ui.text(this, String.format(Locale.KOREA, "+%,d원", py.optInt("amount")), 15, Ui.OK, true)
                    : Ui.chip(this, String.format(Locale.KOREA, "%,d원 대기", py.optInt("amount")), Ui.ACCENT_INK, Ui.ACCENT_SOFT);
            list.addView(Ui.row(this, sent ? Icon.CHECK : Icon.STAR, sent ? Ui.OK : Ui.ACCENT_INK, sent ? Ui.OK_SOFT : Ui.ACCENT_SOFT,
                    st.optString("title", "녹화"),
                    (id.length() >= 8 ? id.substring(0, 4) + "." + id.substring(4, 6) + "." + id.substring(6, 8) : id)
                            + (sent ? " · " + py.optString("label") : " · 계좌 연결 대기"), right, null));
            n++;
        }
        if (n == 0) list.addView(Ui.row(this, Icon.LIST, Ui.SUB, Ui.FIELD, "아직 입금된 보상이 없어요", "미션이 승인되면 여기로 들어와요", null, null));
        Ui.add(this, body, list);
    }

    private String label() {
        return Store.bank(this) + " " + Store.maskedAccount(this).replace("••••", "****");
    }

    private void paint(LinearLayout chips) {
        for (int i = 0; i < chips.getChildCount(); i++) {
            TextView c = (TextView) chips.getChildAt(i);
            boolean on = c.getText().toString().equals(bank);
            c.setTextColor(on ? android.graphics.Color.WHITE : Ui.SUB);
            c.setBackground(Ui.round(on ? Ui.PRIMARY : Ui.FIELD, Ui.dp(this, 99)));
        }
    }

    private void save(String n) {
        if (n.length() < 10 || n.length() > 16) {
            Toast.makeText(this, "계좌번호를 확인해 주세요", Toast.LENGTH_SHORT).show();
            return;
        }
        Store.setPayout(this, bank, n);
        editing = false;
        String lb = label();
        new Thread(() -> {
            String msg;
            try {
                JSONObject r = Uploader.callJson(this, "POST", "/api/payout/account", new JSONObject().put("label", lb));
                int sent = r.optInt("sent_now");
                msg = sent > 0 ? "기다리던 보상 " + sent + "건을 보냈어요" : "다음 보상부터 바로 입금돼요";
                Uploader.refreshPending(this);
            } catch (Exception e) {
                msg = "이 폰에 저장했어요 · 서버에 알리지 못함";
            }
            String m = msg;
            runOnUiThread(() -> { Toast.makeText(this, m, Toast.LENGTH_LONG).show(); onResume(); Payout.checkNew(this); });
        }).start();
    }
}
