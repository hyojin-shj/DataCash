package com.datacache.app;

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
 * 보상 받을 계좌: 받은 보상 · 은행 고르기 + 계좌번호 · 입금 내역(승인된 녹화).
 * 계좌번호는 이 폰에만 저장한다 (서버로 보내지 않음). 시험 버전이라 실제 송금은 하지 않는다.
 */
public class PayoutActivity extends Activity {
    private static final String[] BANKS = {"KB국민", "신한", "우리", "하나", "NH농협", "카카오뱅크", "토스뱅크", "IBK기업"};
    private String bank;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        LinearLayout body = Ui.page(this, "보상 받을 계좌", true);
        bank = Store.bank(this).isEmpty() ? BANKS[0] : Store.bank(this);

        LinearLayout top = Ui.card(this);
        top.addView(Ui.text(this, "받은 보상", 13, Ui.SUB, false));
        TextView won = Ui.text(this, String.format(Locale.KOREA, "%,d원", Store.earned(this)), 28, Ui.INK, true);
        won.setPadding(0, Ui.dp(this, 8), 0, Ui.dp(this, 8));
        top.addView(won);
        top.addView(Ui.text(this, Store.account(this).isEmpty() ? "아직 계좌를 연결하지 않았어요"
                : Store.bank(this) + " " + Store.maskedAccount(this) + " 으로 받아요", 13, Ui.SUB, false));
        Ui.add(this, body, top);

        body.addView(Ui.sectionTitle(this, "입금받을 계좌"));
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
        EditText num = Ui.input(this, "계좌번호 (- 없이)", Store.account(this), InputType.TYPE_CLASS_NUMBER);
        acc.addView(num, Ui.margins(this, -1, -2, 0, 0, 14, 0, 14));
        acc.addView(Ui.button(this, "계좌 연결하고 바로 받기", true, v -> {
            String n = num.getText().toString().replaceAll("[^0-9]", "");
            if (n.length() < 8) {
                Toast.makeText(this, "계좌번호를 확인해 주세요", Toast.LENGTH_SHORT).show();
                return;
            }
            Store.setPayout(this, bank, n);
            Toast.makeText(this, bank + " " + Store.maskedAccount(this) + " 연결했어요", Toast.LENGTH_SHORT).show();
            recreate();
        }));
        Ui.add(this, body, acc);
        TextView note = Ui.text(this, "계좌번호는 이 폰에만 저장되고 서버로 보내지 않아요. 시험 버전이라 실제 송금 대신 적립으로 기록돼요.", 12, Ui.SUB, false);
        note.setPadding(Ui.dp(this, 4), 0, Ui.dp(this, 4), Ui.dp(this, 6));
        body.addView(note);

        body.addView(Ui.sectionTitle(this, "입금 내역"));
        LinearLayout list = Ui.listCard(this);
        int n = 0;
        for (File d : Store.sessions(this)) {
            JSONObject st = Uploader.localState(d);
            if (!st.optString("state").equals("done") || st.optInt("reward") <= 0) continue;
            String id = d.getName();
            list.addView(Ui.row(this, Icon.CHECK, Ui.OK, Ui.OK_SOFT, st.optString("title", "녹화"),
                    id.length() >= 8 ? id.substring(0, 4) + "." + id.substring(4, 6) + "." + id.substring(6, 8) : id,
                    Ui.text(this, String.format(Locale.KOREA, "+%,d원", st.optInt("reward")), 15, Ui.OK, true), null));
            n++;
        }
        if (n == 0) list.addView(Ui.row(this, Icon.STAR, Ui.SUB, Ui.FIELD, "아직 입금된 보상이 없어요", "미션이 승인되면 여기로 들어와요", null, null));
        Ui.add(this, body, list);
    }

    private void paint(LinearLayout chips) {
        for (int i = 0; i < chips.getChildCount(); i++) {
            TextView c = (TextView) chips.getChildAt(i);
            boolean on = c.getText().toString().equals(bank);
            c.setTextColor(on ? android.graphics.Color.WHITE : Ui.SUB);
            c.setBackground(Ui.round(on ? Ui.PRIMARY : Ui.FIELD, Ui.dp(this, 99)));
        }
    }
}
