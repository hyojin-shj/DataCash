package com.datacash.app;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.text.InputType;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.TextView;

import org.json.JSONObject;

import java.net.HttpURLConnection;
import java.net.URL;

/**
 * 스마트 기기 연결: 데이터 서버(주소·토큰, 저장하고 연결 확인) + 워치(갤럭시 워치에 데이터캐시 워치 앱을 켜 두면
 * 폰 녹화에 자동으로 합류 — 폰과 워치는 서버를 거쳐 시계를 맞춘다) + 쥐는 세기 맞추기.
 */
public class DevicesActivity extends BaseActivity {
    private TextView status;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        LinearLayout body = Ui.page(this, getString(R.string.menu_devices), true, Lang.button(this));
        TextView intro = Ui.text(this, getString(R.string.devices_intro),
                13, Ui.SUB, false);
        intro.setPadding(Ui.dp(this, 4), 0, Ui.dp(this, 4), Ui.dp(this, 14));
        body.addView(intro);

        LinearLayout lang = Ui.listCard(this);             // 앱 언어 (머리줄 단추와 같은 창)
        String[] langNames = getResources().getStringArray(R.array.lang_names);
        lang.addView(Ui.row(this, Icon.LIST, Ui.VIOLET, Ui.VIOLET_SOFT, getString(R.string.lang_section), langNames[Lang.index(this)],
                null, v -> Lang.pick(this)));
        Ui.add(this, body, lang);

        LinearLayout srv = Ui.card(this);
        srv.addView(Ui.row(this, Icon.CLOUD, Ui.PRIMARY, Ui.PRIMARY_SOFT, getString(R.string.data_server), getString(R.string.data_server_desc), null, null),
                Ui.margins(this, -1, -2, 0, -12, -12, -12, 8));
        EditText url = Ui.input(this, getString(R.string.server_hint), Store.server(this), InputType.TYPE_TEXT_VARIATION_URI);
        srv.addView(url, Ui.margins(this, -1, -2, 0, 0, 0, 0, 10));
        EditText tok = Ui.input(this, getString(R.string.token_hint), Store.token(this),
                InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_VISIBLE_PASSWORD);
        srv.addView(tok, Ui.margins(this, -1, -2, 0, 0, 0, 0, 14));
        srv.addView(Ui.button(this, getString(R.string.save_and_check), true, v -> {
            Store.setServer(this, url.getText().toString(), tok.getText().toString());
            check();
        }));
        srv.addView(Ui.button(this, getString(R.string.auto_find), false, v -> {
            status.setText(R.string.finding);
            new Thread(() -> {
                String found = Discover.find(this);
                runOnUiThread(() -> {
                    if (found == null) {
                        status.setText(R.string.not_found);
                        status.setTextColor(Ui.PRIMARY_DARK);
                        return;
                    }
                    url.setText(found);
                    Store.setServer(this, found, tok.getText().toString());
                    check();
                });
            }).start();
        }), Ui.margins(this, -1, -2, 0, 8, 0, 0, 0));
        status = Ui.text(this, "", 13, Ui.SUB, false);
        status.setPadding(Ui.dp(this, 4), Ui.dp(this, 10), 0, 0);
        srv.addView(status);
        Ui.add(this, body, srv);

        LinearLayout watch = Ui.card(this);
        watch.addView(Ui.row(this, Icon.WATCH, Ui.ACCENT_INK, Ui.ACCENT_SOFT, getString(R.string.galaxy_watch), getString(R.string.watch_desc), null, null),
                Ui.margins(this, -1, -2, 0, -12, -12, -12, 8));
        watch.addView(Ui.text(this, getString(R.string.watch_steps), 13, Ui.SUB, false));
        watch.addView(Ui.button(this, getString(R.string.calib_watch_btn), true, v -> startActivity(new Intent(this, CalibrationActivity.class))),
                Ui.margins(this, -1, -2, 0, 0, 16, 0, 0));
        Ui.add(this, body, watch);

        body.addView(Ui.sectionTitle(this, getString(R.string.coming_soon)));
        LinearLayout soon = Ui.listCard(this);
        soon.addView(Ui.row(this, Icon.FILM, Ui.SUB, Ui.FIELD, getString(R.string.home_camera), getString(R.string.home_camera_desc), Ui.chip(this, getString(R.string.preparing), Ui.SUB, Ui.FIELD), null));
        soon.addView(Ui.row(this, Icon.SPARK, Ui.SUB, Ui.FIELD, getString(R.string.speaker_hub), getString(R.string.speaker_hub_desc), Ui.chip(this, getString(R.string.preparing), Ui.SUB, Ui.FIELD), null));
        Ui.add(this, body, soon);
        check();
    }

    private void check() {
        status.setText(R.string.checking);
        new Thread(() -> {
            String msg;
            try {
                long a = Clock.now();
                HttpURLConnection h = (HttpURLConnection) new URL(Store.server(this) + "/api/time").openConnection();
                h.setRequestProperty("X-DC-Token", Store.token(this));
                h.setRequestProperty("X-DC-Lang", Store.lang(this));
                h.setConnectTimeout(2000);
                h.setReadTimeout(2000);
                int code = h.getResponseCode();
                if (code == 200) {
                    new JSONObject(new String(h.getInputStream().readAllBytes(), "UTF-8"));
                    msg = getString(R.string.connected_rtt, (Clock.now() - a) / 1e6);
                } else {
                    msg = code == 401 ? getString(R.string.token_wrong) : getString(R.string.server_code, code);
                }
            } catch (Exception e) {
                msg = getString(R.string.not_connected_check);
            }
            String m = msg;
            runOnUiThread(() -> {
                status.setText(m);
                status.setTextColor(m.startsWith("●") ? Ui.OK : Ui.PRIMARY_DARK);
            });
        }).start();
    }
}
