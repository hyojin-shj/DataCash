package com.datacache.app;

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
 * 폰 녹화에 자동으로 합류 — 폰과 워치는 서버를 거쳐 시계를 맞춘다) + 쥐는 힘 맞추기.
 */
public class DevicesActivity extends Activity {
    private TextView status;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        LinearLayout body = Ui.page(this, "스마트 기기 연결", true);
        TextView intro = Ui.text(this, "기기를 연결하면 영상과 손목 움직임·쥐는 힘·심박이 같은 시간축으로 맞춰진 고품질 등급이 되어 보상이 더 높아요.",
                13, Ui.SUB, false);
        intro.setPadding(Ui.dp(this, 4), 0, Ui.dp(this, 4), Ui.dp(this, 14));
        body.addView(intro);

        LinearLayout srv = Ui.card(this);
        srv.addView(Ui.row(this, Icon.CLOUD, Ui.PRIMARY, Ui.PRIMARY_SOFT, "데이터 서버", "녹화가 끝나면 여기로 올라가 검수돼요", null, null),
                Ui.margins(this, -1, -2, 0, -12, -12, -12, 8));
        EditText url = Ui.input(this, "서버 주소 (예: http://192.168.0.10:8800)", Store.server(this), InputType.TYPE_TEXT_VARIATION_URI);
        srv.addView(url, Ui.margins(this, -1, -2, 0, 0, 0, 0, 10));
        EditText tok = Ui.input(this, "토큰 (서버 화면에 나온 값)", Store.token(this),
                InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_VISIBLE_PASSWORD);
        srv.addView(tok, Ui.margins(this, -1, -2, 0, 0, 0, 0, 14));
        srv.addView(Ui.button(this, "저장하고 연결 확인", true, v -> {
            Store.setServer(this, url.getText().toString(), tok.getText().toString());
            check();
        }));
        status = Ui.text(this, "", 13, Ui.SUB, false);
        status.setPadding(Ui.dp(this, 4), Ui.dp(this, 10), 0, 0);
        srv.addView(status);
        Ui.add(this, body, srv);

        LinearLayout watch = Ui.card(this);
        watch.addView(Ui.row(this, Icon.WATCH, Ui.ACCENT_INK, Ui.ACCENT_SOFT, "갤럭시 워치", "손목 움직임 · 능동 진동 촉각 · 심박", null, null),
                Ui.margins(this, -1, -2, 0, -12, -12, -12, 8));
        watch.addView(Ui.text(this, "1. 워치에 데이터캐시 워치 앱을 깔고 켜 두세요 (같은 서버 주소)\n"
                + "2. 폰에서 녹화를 시작하면 워치 기록이 자동으로 같이 시작돼요\n"
                + "3. 끝나면 워치가 알아서 올리고, 서버가 폰과 시계를 맞춰요 — 둘을 직접 이을 필요 없어요", 13, Ui.SUB, false));
        watch.addView(Ui.button(this, "워치로 쥐는 힘 맞추기 (한 번만)", true, v -> startActivity(new Intent(this, CalibrationActivity.class))),
                Ui.margins(this, -1, -2, 0, 0, 16, 0, 0));
        Ui.add(this, body, watch);

        body.addView(Ui.sectionTitle(this, "곧 연결할 수 있어요"));
        LinearLayout soon = Ui.listCard(this);
        soon.addView(Ui.row(this, Icon.FILM, Ui.SUB, Ui.FIELD, "홈 카메라", "고정된 각도에서 함께 찍어 3인칭 장면을 더해요", Ui.chip(this, "준비 중", Ui.SUB, Ui.FIELD), null));
        soon.addView(Ui.row(this, Icon.SPARK, Ui.SUB, Ui.FIELD, "스마트 스피커·허브", "집 안 소리·기기 상태를 같은 시간축에", Ui.chip(this, "준비 중", Ui.SUB, Ui.FIELD), null));
        Ui.add(this, body, soon);
        check();
    }

    private void check() {
        status.setText("연결 확인 중…");
        new Thread(() -> {
            String msg;
            try {
                long a = Clock.now();
                HttpURLConnection h = (HttpURLConnection) new URL(Store.server(this) + "/api/time").openConnection();
                h.setRequestProperty("X-DC-Token", Store.token(this));
                h.setConnectTimeout(2000);
                h.setReadTimeout(2000);
                int code = h.getResponseCode();
                if (code == 200) {
                    new JSONObject(new String(h.getInputStream().readAllBytes(), "UTF-8"));
                    msg = String.format(java.util.Locale.KOREA, "● 연결됨 · 왕복 %.0fms", (Clock.now() - a) / 1e6);
                } else {
                    msg = code == 401 ? "토큰이 맞지 않아요" : "서버 응답 " + code;
                }
            } catch (Exception e) {
                msg = "연결 안 됨 · 같은 와이파이인지, 주소가 맞는지 확인해 주세요";
            }
            String m = msg;
            runOnUiThread(() -> {
                status.setText(m);
                status.setTextColor(m.startsWith("●") ? Ui.OK : Ui.PRIMARY_DARK);
            });
        }).start();
    }
}
