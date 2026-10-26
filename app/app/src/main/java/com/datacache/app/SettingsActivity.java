package com.datacache.app;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.text.InputType;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.Toast;

/** 서버 주소·토큰 (같은 와이파이의 서버, 시작할 때 터미널에 나온 토큰), 동의 다시 보기. */
public class SettingsActivity extends Activity {
    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        LinearLayout body = Ui.page(this, "설정", true);
        LinearLayout c = Ui.card(this);
        c.addView(Ui.text(this, "서버 주소", 14, Ui.SUB, false));
        EditText url = new EditText(this);
        url.setText(Store.server(this));
        url.setInputType(InputType.TYPE_TEXT_VARIATION_URI);
        c.addView(url);
        c.addView(Ui.text(this, "토큰", 14, Ui.SUB, false));
        EditText tok = new EditText(this);
        tok.setText(Store.token(this));
        tok.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_VISIBLE_PASSWORD);
        c.addView(tok);
        Ui.add(this, body, c);
        Ui.add(this, body, Ui.button(this, "저장", true, v -> {
            Store.setServer(this, url.getText().toString(), tok.getText().toString());
            Toast.makeText(this, "저장했어요", Toast.LENGTH_SHORT).show();
            finish();
        }));
        Ui.add(this, body, Ui.button(this, "동의 다시 보기", false, v -> startActivity(new Intent(this, ConsentActivity.class))));
    }
}
