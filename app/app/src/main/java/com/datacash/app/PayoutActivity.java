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
public class PayoutActivity extends BaseActivity {
    private String[] BANKS;      // 저장 값 (R.array.bank_values) — 보이는 이름은 R.array.banks
    private String bank;

    /** 저장된 은행 값 → 지금 언어의 이름. */
    static String bankLabel(android.content.Context c, String value) {
        String[] vals = c.getResources().getStringArray(R.array.bank_values), names = c.getResources().getStringArray(R.array.banks);
        for (int i = 0; i < vals.length && i < names.length; i++) if (vals[i].equals(value)) return names[i];
        return value;
    }
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
        total.setText(Ui.won(this, r.optInt("total")));
        int waiting = r.optInt("waiting");
        if (waiting > 0) status.setText(getString(R.string.waiting_status, waiting));
    }

    private void render() {
        BANKS = getResources().getStringArray(R.array.bank_values);
        String[] bankNames = getResources().getStringArray(R.array.banks);
        LinearLayout body = Ui.page(this, getString(R.string.menu_payout), true);
        bank = Store.bank(this).isEmpty() ? BANKS[0] : Store.bank(this);
        LinearLayout top = Ui.card(this);
        top.addView(Ui.text(this, getString(R.string.received), 13, Ui.SUB, false));
        total = Ui.text(this, "-", 28, Ui.INK, true);
        total.setPadding(0, Ui.dp(this, 8), 0, Ui.dp(this, 8));
        top.addView(total);
        status = Ui.text(this, Store.account(this).isEmpty() ? getString(R.string.no_account) : getString(R.string.receive_to, label()), 13, Ui.SUB, false);
        top.addView(status);
        Ui.add(this, body, top);

        body.addView(Ui.sectionTitle(this, getString(R.string.deposit_account)));
        if (!Store.account(this).isEmpty() && !editing) {
            LinearLayout c = Ui.listCard(this);
            TextView change = Ui.chip(this, getString(R.string.change_account), Ui.SUB, Ui.FIELD);
            change.setOnClickListener(v -> { editing = true; render(); });
            c.addView(Ui.row(this, Icon.STAR, Ui.ACCENT_INK, Ui.ACCENT_SOFT, bankLabel(this, Store.bank(this)), getString(R.string.linked_line, Store.maskedAccount(this)), change, null));
            Ui.add(this, body, c);
        } else {
            LinearLayout acc = Ui.card(this);
            HorizontalScrollView hs = new HorizontalScrollView(this);
            hs.setHorizontalScrollBarEnabled(false);
            LinearLayout chips = new LinearLayout(this);
            hs.addView(chips);
            for (int i = 0; i < BANKS.length; i++) {
                String b = BANKS[i];
                TextView c = Ui.chip(this, i < bankNames.length ? bankNames[i] : b, 0, 0);
                c.setTag(b);
                c.setTextSize(14);
                c.setPadding(Ui.dp(this, 14), Ui.dp(this, 9), Ui.dp(this, 14), Ui.dp(this, 9));
                c.setOnClickListener(v -> { bank = b; paint(chips); });
                chips.addView(c, Ui.margins(this, -2, -2, 0, 0, 0, 8, 0));
            }
            paint(chips);
            acc.addView(hs);
            EditText num = Ui.input(this, getString(R.string.account_hint), "", InputType.TYPE_CLASS_NUMBER);
            acc.addView(num, Ui.margins(this, -1, -2, 0, 0, 14, 0, 14));
            acc.addView(Ui.button(this, getString(R.string.link_and_receive), true, v -> save(num.getText().toString().replaceAll("[^0-9]", ""))));
            Ui.add(this, body, acc);
        }
        TextView note = Ui.text(this, getString(R.string.payout_note), 12, Ui.SUB, false);
        note.setPadding(Ui.dp(this, 4), 0, Ui.dp(this, 4), Ui.dp(this, 6));
        body.addView(note);

        body.addView(Ui.sectionTitle(this, getString(R.string.payout_history)));
        LinearLayout list = Ui.listCard(this);
        int n = 0;
        for (File d : Store.sessions(this)) {
            JSONObject st = Uploader.localState(d);
            JSONObject py = st.optJSONObject("payout");
            if (py == null) continue;
            String id = d.getName();
            boolean sent = py.optString("status").equals("sent");
            TextView right = sent ? Ui.text(this, Ui.plusWon(this, py.optInt("amount")), 15, Ui.OK, true)
                    : Ui.chip(this, getString(R.string.waiting_chip, py.optInt("amount")), Ui.ACCENT_INK, Ui.ACCENT_SOFT);
            list.addView(Ui.row(this, sent ? Icon.CHECK : Icon.STAR, sent ? Ui.OK : Ui.ACCENT_INK, sent ? Ui.OK_SOFT : Ui.ACCENT_SOFT,
                    st.optString("title", getString(R.string.recording)),
                    (id.length() >= 8 ? id.substring(0, 4) + "." + id.substring(4, 6) + "." + id.substring(6, 8) : id)
                            + (sent ? " · " + py.optString("label") : " · " + getString(R.string.waiting_account)), right, null));
            n++;
        }
        if (n == 0) list.addView(Ui.row(this, Icon.LIST, Ui.SUB, Ui.FIELD, getString(R.string.no_payouts), getString(R.string.no_payouts_desc), null, null));
        Ui.add(this, body, list);
    }

    private String label() {
        return bankLabel(this, Store.bank(this)) + " " + Store.maskedAccount(this).replace("••••", "****");
    }

    private void paint(LinearLayout chips) {
        for (int i = 0; i < chips.getChildCount(); i++) {
            TextView c = (TextView) chips.getChildAt(i);
            boolean on = String.valueOf(c.getTag()).equals(bank);
            c.setTextColor(on ? android.graphics.Color.WHITE : Ui.SUB);
            c.setBackground(Ui.round(on ? Ui.PRIMARY : Ui.FIELD, Ui.dp(this, 99)));
        }
    }

    private void save(String n) {
        if (n.length() < 10 || n.length() > 16) {
            Toast.makeText(this, R.string.check_account, Toast.LENGTH_SHORT).show();
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
                msg = sent > 0 ? getString(R.string.sent_waiting, sent) : getString(R.string.next_paid);
                Uploader.refreshPending(this);
            } catch (Exception e) {
                msg = getString(R.string.saved_local);
            }
            String m = msg;
            runOnUiThread(() -> { Toast.makeText(this, m, Toast.LENGTH_LONG).show(); onResume(); Payout.checkNew(this); });
        }).start();
    }
}
