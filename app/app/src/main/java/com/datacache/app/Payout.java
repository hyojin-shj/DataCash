package com.datacache.app;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Color;
import android.media.AudioAttributes;
import android.media.AudioFormat;
import android.media.AudioTrack;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.view.Gravity;
import android.view.animation.OvershootInterpolator;
import android.widget.LinearLayout;
import android.widget.TextView;

import org.json.JSONObject;

import java.io.File;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;

/**
 * 보상 입금 알림: 녹화 상태의 payout 이 새로 'sent' 가 되면 '띠링!' (동전 · +N원 커지는 숫자 · 두 음 소리 · 짧은 진동),
 * 계좌가 없어 'waiting_account' 가 되면 '보상 확정 · 계좌를 연결하면 바로 입금돼요'. 녹화마다 한 번씩만.
 */
final class Payout {
    private Payout() {}

    private static SharedPreferences p(Context c) { return c.getSharedPreferences("payout", Context.MODE_PRIVATE); }

    private static Set<String> set(Context c, String k) { return new HashSet<>(p(c).getStringSet(k, new HashSet<>())); }

    private static void add(Context c, String k, Set<String> more) {
        Set<String> s = set(c, k);
        s.addAll(more);
        p(c).edit().putStringSet(k, s).apply();
    }

    /** 새로 들어온 입금·대기를 한 번에 모아 팝업. 화면이 떠 있을 때 부른다. */
    static void checkNew(Activity a) {
        Set<String> celebrated = set(a, "celebrated"), told = set(a, "told_waiting");
        Set<String> sent = new HashSet<>(), waiting = new HashSet<>();
        int won = 0, wait = 0;
        String title = null, label = "";
        for (File d : Store.sessions(a)) {
            JSONObject st = Uploader.localState(d);
            JSONObject py = st.optJSONObject("payout");
            if (py == null) continue;
            String id = d.getName();
            if (py.optString("status").equals("sent") && !celebrated.contains(id)) {
                sent.add(id);
                won += py.optInt("amount");
                title = st.optString("title", "미션");
                label = py.optString("label");
            } else if (py.optString("status").equals("waiting_account") && !told.contains(id)) {
                waiting.add(id);
                wait += py.optInt("amount");
            }
        }
        if (!sent.isEmpty()) {
            add(a, "celebrated", sent);
            ding(a, won, sent.size() == 1 ? title + " 미션 보상이 입금됐어요" : sent.size() + "건 미션 보상이 입금됐어요", label);
        } else if (!waiting.isEmpty()) {
            add(a, "told_waiting", waiting);
            confirmed(a, wait);
        }
    }

    private static void ding(Activity a, int won, String what, String label) {
        if (a.isFinishing()) return;
        LinearLayout box = new LinearLayout(a);
        box.setOrientation(LinearLayout.VERTICAL);
        box.setGravity(Gravity.CENTER_HORIZONTAL);
        box.setPadding(Ui.dp(a, 24), Ui.dp(a, 28), Ui.dp(a, 24), Ui.dp(a, 8));
        TextView coin = Ui.text(a, "₩", 34, Color.WHITE, true);
        coin.setGravity(Gravity.CENTER);
        coin.setBackground(Ui.gradient(Ui.dp(a, 36)));
        box.addView(coin, new LinearLayout.LayoutParams(Ui.dp(a, 72), Ui.dp(a, 72)));
        TextView t = Ui.text(a, "띠링!", 18, Ui.ACCENT_INK, true);
        t.setPadding(0, Ui.dp(a, 14), 0, 0);
        box.addView(t);
        TextView amount = Ui.text(a, String.format(Locale.KOREA, "+%,d원", won), 34, Ui.INK, true);
        amount.setPadding(0, Ui.dp(a, 4), 0, Ui.dp(a, 6));
        amount.setScaleX(0.4f);
        amount.setScaleY(0.4f);
        box.addView(amount);
        TextView w = Ui.text(a, what, 15, Ui.INK, false);
        w.setGravity(Gravity.CENTER);
        box.addView(w);
        if (!label.isEmpty()) {
            TextView l = Ui.text(a, label + " 으로 받았어요", 13, Ui.SUB, false);
            l.setPadding(0, Ui.dp(a, 6), 0, 0);
            box.addView(l);
        }
        new AlertDialog.Builder(a).setView(box).setPositiveButton("좋아요", null).show();
        amount.animate().scaleX(1f).scaleY(1f).setDuration(450).setInterpolator(new OvershootInterpolator(2.2f)).start();
        sound(a);
    }

    private static void confirmed(Activity a, int won) {
        if (a.isFinishing()) return;
        new AlertDialog.Builder(a).setTitle(String.format(Locale.KOREA, "보상 확정 +%,d원", won))
                .setMessage("계좌를 연결하면 바로 입금돼요")
                .setPositiveButton("계좌 연결하기", (d, x) -> a.startActivity(new Intent(a, PayoutActivity.class)))
                .setNegativeButton("나중에", null).show();
    }

    /** 두 음 '띠링' (B5 짧게 → E6 길게) + 짧은 진동 두 번. */
    static void sound(Context c) {
        new Thread(() -> {
            int rate = 44100;
            short[] pcm = new short[(int) (rate * 0.55)];
            int a = (int) (rate * 0.07);
            for (int i = 0; i < pcm.length; i++) {
                double f = i < a ? 987.77 : 1318.51, t = (double) (i < a ? i : i - a) / rate;
                double env = i < a ? Math.min(1, t * 200) * (1 - t / 0.07 * 0.3) : Math.exp(-t * 6);
                pcm[i] = (short) (Math.sin(2 * Math.PI * f * t) * env * 9000);
            }
            try {
                AudioTrack tr = new AudioTrack.Builder()
                        .setAudioAttributes(new AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_NOTIFICATION_EVENT).build())
                        .setAudioFormat(new AudioFormat.Builder().setSampleRate(rate).setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                                .setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
                        .setTransferMode(AudioTrack.MODE_STATIC).setBufferSizeInBytes(pcm.length * 2).build();
                tr.write(pcm, 0, pcm.length);
                tr.play();
                Thread.sleep(700);
                tr.release();
            } catch (Exception ignored) {
            }
        }).start();
        Vibrator v = (Vibrator) c.getSystemService(Context.VIBRATOR_SERVICE);
        if (v != null) v.vibrate(VibrationEffect.createWaveform(new long[]{0, 40, 60, 90}, -1));
    }
}
