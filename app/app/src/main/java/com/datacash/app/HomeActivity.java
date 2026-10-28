package com.datacash.app;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.text.SpannableString;
import android.text.style.ForegroundColorSpan;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.View;
import android.widget.FrameLayout;
import android.widget.HorizontalScrollView;
import android.widget.LinearLayout;
import android.widget.TextView;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * 첫 화면.
 *   모은 보상        크게 + 사물에 빗댄 한 줄 ("치킨 한 마리 값을 벌었어요!"), 다음 목표는 '미션 몇 개'로
 *   오늘의 보너스 미션 하루에 하나 +50% (서버와 같은 식으로 고름). 진행 고리 가운데 빨간 버튼 → 아직 안 찍은 다음 상황 촬영
 *   기업 의뢰 미션    옆으로 넘기는 카드: 보상 · 의뢰 목적 · 찍은 상황 수 · 상황 골라서 찍기
 *   더 많은 기능      보상 받을 계좌 · 스마트 기기 연결 · 쥐는 세기 맞추기 · 내 녹화 기록 · 동의 다시 보기
 * 처음이면 동의 화면부터.
 */
public class HomeActivity extends Activity {
    private final Map<String, Integer> scarce = new HashMap<>();   // 서버가 알려 주는 부족한 미션 보너스 %
    private String todayId;
    private LinearLayout pages, dots;
    private HorizontalScrollView pager;

    private final android.os.Handler ui = new android.os.Handler(android.os.Looper.getMainLooper());
    private boolean visible;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        if (!Store.consented(this)) startActivity(new Intent(this, ConsentActivity.class));
        if (android.os.Build.VERSION.SDK_INT >= 33
                && checkSelfPermission(android.Manifest.permission.POST_NOTIFICATIONS) != android.content.pm.PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{android.Manifest.permission.POST_NOTIFICATIONS}, 11);   // 적립 알림
        }
        for (File d : Store.sessions(this)) {     // 앱이 꺼져서 멈춘 올리기를 다시 맡긴다
            String st = Uploader.localState(d).optString("state");
            if ((st.equals("uploading") || st.equals("saved")) && new File(d, "meta.json").exists()) {
                Uploader.save(d, "state", "saved");
                ProcessService.enqueue(this, d);
            }
        }
    }

    /** 떠 있는 동안 5초마다 처리 중인 녹화의 결과를 받고, 입금되면 '띠링'. */
    private void poll() {
        if (!visible) return;
        new Thread(() -> {
            String before = Uploader.signature(this);
            Uploader.refreshPending(this);
            boolean changed = !before.equals(Uploader.signature(this));
            ui.post(() -> {
                if (!visible) return;
                if (changed) render();          // 바뀐 게 있을 때만 다시 그린다 (스크롤이 튀지 않게)
                Payout.checkNew(this);
                ui.postDelayed(this::poll, 5000);
            });
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
        ProcessService.resume(this);
        ui.postDelayed(this::poll, 1500);
        render();
        new Thread(() -> {
            try {
                Discover.ensure(this);           // 맥 IP 가 바뀌었으면 서버를 다시 찾는다
                JSONArray ms = Uploader.missions(this);
                Map<String, Integer> b = new HashMap<>();
                String t = null;
                for (int i = 0; i < ms.length(); i++) {
                    JSONObject m = ms.getJSONObject(i);
                    if (m.optInt("bonus_pct") > 0) b.put(m.getString("id"), m.getInt("bonus_pct"));
                    if (m.optBoolean("today")) t = m.getString("id");
                }
                String today = t;
                boolean fresh = Uploader.changed;
                runOnUiThread(() -> {
                    boolean same = scarce.equals(b) && (today == null || today.equals(todayId)) && !fresh;
                    scarce.clear();
                    scarce.putAll(b);
                    if (today != null) todayId = today;
                    if (!same) render();          // 의뢰 미션이 새로 열렸거나 보너스가 바뀌었을 때만 다시 그린다
                });
            } catch (Exception ignored) {   // 서버가 없으면 폰 안의 미션 목록만으로
            }
        }).start();
    }

    private void render() {
        int todayCount = 0, todayWon = 0;
        String todayPrefix = Clock.sessionId().substring(0, 8);
        for (File d : Store.sessions(this)) {
            if (!d.getName().startsWith(todayPrefix)) continue;
            todayCount++;
            JSONObject st = Uploader.localState(d);
            if (st.optString("state").equals("done")) todayWon += st.optInt("reward") + st.optInt("appeal_extra");
        }
        TextView badge = Ui.chip(this, todayWon > 0 ? String.format(Locale.KOREA, "오늘 +%,d원", todayWon)
                : todayCount == 0 ? "오늘 첫 미션" : "오늘 " + todayCount + "번 찍음", Ui.PRIMARY, Ui.PRIMARY_SOFT);
        badge.setTextSize(13);
        LinearLayout body = Ui.page(this, null, false, badge);
        try {
            if (todayId == null) todayId = Store.todayMission(this);
            Ui.add(this, body, earnings());
            Ui.add(this, body, todayCard(Store.mission(this, todayId)));
            LinearLayout sec = new LinearLayout(this);
            sec.setGravity(Gravity.CENTER_VERTICAL);
            sec.addView(Ui.sectionTitle(this, "기업 의뢰 미션"), Ui.weight());
            TextView hint = Ui.text(this, "옆으로 넘겨 보세요", 13, Ui.SUB, false);
            hint.setPadding(0, Ui.dp(this, 12), Ui.dp(this, 4), Ui.dp(this, 10));
            sec.addView(hint);
            body.addView(sec);
            body.addView(pager());
            dots = new LinearLayout(this);
            dots.setGravity(Gravity.CENTER);
            body.addView(dots, Ui.margins(this, -1, -2, 0, 0, 12, 0, 6));
            showDots(0);
        } catch (Exception e) {
            body.addView(Ui.text(this, "미션을 읽지 못했어요: " + e.getMessage(), 13, Ui.SUB, false));
        }

        body.addView(Ui.sectionTitle(this, "더 많은 기능"));
        LinearLayout menu = Ui.listCard(this);
        menu.addView(Ui.row(this, Icon.STAR, Ui.ACCENT_INK, Ui.ACCENT_SOFT, "보상 받을 계좌",
                Store.account(this).isEmpty() ? "계좌를 연결하면 미션마다 바로 입금" : Store.bank(this) + " " + Store.maskedAccount(this),
                null, v -> open(PayoutActivity.class)));
        menu.addView(Ui.row(this, Icon.WATCH, Ui.PRIMARY, Ui.PRIMARY_SOFT, "스마트 기기 연결",
                "워치를 이으면 촉각·생체까지 담겨 고품질 등급", null, v -> open(DevicesActivity.class)));
        menu.addView(Ui.row(this, Icon.HAND, Ui.PRIMARY, Ui.PRIMARY_SOFT, "쥐는 세기 맞추기",
                "물병을 가볍게·평소처럼·꽉 쥐면 워치가 내 쥐는 세기를 배워요 (한 번만)", null, v -> open(CalibrationActivity.class)));
        menu.addView(Ui.row(this, Icon.PERSON, Ui.OK, Ui.OK_SOFT, "개인정보 동의 인원",
                "함께 찍히는 가족, 동의 받고 추가", null, v -> open(PeopleActivity.class)));
        menu.addView(Ui.row(this, Icon.FILM, Ui.ACCENT_INK, Ui.ACCENT_SOFT, "내 녹화 기록",
                "검수 결과와 보상, 이의 신청", null, v -> open(HistoryActivity.class)));
        menu.addView(Ui.row(this, Icon.LIST, Ui.VIOLET, Ui.VIOLET_SOFT, "이의 신청 현황",
                "자동 확인 · 검토자 확인 · 결과", null, v -> open(AppealsActivity.class)));
        Ui.add(this, body, menu);

        LinearLayout privacy = Ui.listCard(this);
        privacy.addView(Ui.row(this, Icon.SHIELD, Ui.SUB, Ui.FIELD, "내 데이터는 이렇게 지켜져요",
                "동의하지 않은 사람·아이·화면·거울은 가리고, 나와 동의한 사람 얼굴은 실존하지 않는 가상 인물로 바꾼 뒤에만 판매돼요", null, null));
        Ui.add(this, body, privacy);
    }

    private void open(Class<?> c) {
        startActivity(new Intent(this, c));
    }

    // ---------- 모은 보상 ----------

    /** 번 돈 이하인 가장 비싼 것에 빗댄다 (2026년 대략 가격). 숫자보다 사물이 먼저 와닿게. */
    private static final String[] THING = {"아이스크림 하나", "아메리카노 한 잔", "국밥 한 그릇", "영화 한 편", "치킨 한 마리",
            "한 달 휴대폰 요금", "운동화 한 켤레", "무선 이어폰", "제주도 여행"};
    private static final int[] PRICE = {1_500, 4_500, 10_000, 15_000, 22_000, 55_000, 120_000, 300_000, 600_000};

    private LinearLayout earnings() {
        int won = Store.earned(this);
        LinearLayout c = Ui.card(this);
        c.setPadding(Ui.dp(this, 22), Ui.dp(this, 20), Ui.dp(this, 22), Ui.dp(this, 20));
        c.addView(Ui.text(this, "모은 데이터 보상", 14, Ui.SUB, true));
        TextView big = Ui.text(this, String.format(Locale.KOREA, "%,d원", won), 34, Ui.INK, true);
        c.addView(big, Ui.margins(this, -2, -2, 0, 0, 6, 0, 0));
        int i = -1;
        while (i + 1 < PRICE.length && won >= PRICE[i + 1]) i++;
        TextView like = Ui.text(this, "", 17, Ui.INK, true);
        if (i < 0) {
            like.setText("첫 미션이면 아이스크림 하나부터!");
        } else {
            SpannableString s = new SpannableString(THING[i] + " 값을 벌었어요!");
            s.setSpan(new ForegroundColorSpan(Ui.ACCENT_INK), 0, THING[i].length(), 0);
            like.setText(s);
        }
        like.setPadding(0, Ui.dp(this, 8), 0, 0);
        c.addView(like);
        if (i + 1 < PRICE.length) {
            int per = Store.perMission(this);
            int n = (PRICE[i + 1] - won + per - 1) / per;
            TextView sub = Ui.text(this, "미션 " + n + "개 더 하면 " + THING[i + 1] + "!", 13, Ui.SUB, false);
            sub.setPadding(0, Ui.dp(this, 8), 0, 0);
            c.addView(sub);
        }
        c.setOnClickListener(v -> open(PayoutActivity.class));
        return c;
    }

    // ---------- 오늘의 보너스 미션 ----------

    private LinearLayout todayCard(JSONObject m) throws Exception {
        String id = m.getString("id");
        JSONArray sits = m.getJSONArray("situations");
        Set<String> done = Store.doneSituations(this, id);
        LinearLayout c = Ui.card(this);
        c.setGravity(Gravity.CENTER_HORIZONTAL);
        c.setPadding(Ui.dp(this, 20), Ui.dp(this, 22), Ui.dp(this, 20), Ui.dp(this, 20));
        LinearLayout badge = new LinearLayout(this);
        badge.setGravity(Gravity.CENTER_VERTICAL);
        badge.setBackground(Ui.round(Ui.ACCENT_SOFT, Ui.dp(this, 20)));
        badge.setPadding(Ui.dp(this, 10), Ui.dp(this, 5), Ui.dp(this, 12), Ui.dp(this, 5));
        badge.addView(new Icon(this, Icon.STAR, Ui.ACCENT_INK), new LinearLayout.LayoutParams(Ui.dp(this, 14), Ui.dp(this, 14)));
        TextView bt = Ui.text(this, "오늘의 보너스 미션 · 보상 +50%", 12, Ui.ACCENT_INK, true);
        bt.setPadding(Ui.dp(this, 5), 0, 0, 0);
        badge.addView(bt);
        c.addView(badge, Ui.wrap());
        TextView title = Ui.text(this, m.getString("title"), 30, Ui.INK, true);
        title.setPadding(0, Ui.dp(this, 12), 0, Ui.dp(this, 6));
        c.addView(title, Ui.wrap());
        TextView guide = Ui.text(this, m.getString("guide"), 14, Ui.SUB, false);
        guide.setGravity(Gravity.CENTER);
        c.addView(guide, Ui.wrap());

        RecordRing ring = new RecordRing(this);
        ring.setProgress(sits.length() == 0 ? 0f : (float) done.size() / sits.length());
        ring.setContentDescription("녹화 시작");
        int next = 0;
        while (next < sits.length() && done.contains(sits.getString(next))) next++;
        int nextIdx = next >= sits.length() ? 0 : next;
        ring.setOnClickListener(v -> record(id, sits.optString(nextIdx)));
        c.addView(ring, Ui.margins(this, Ui.dp(this, 200), Ui.dp(this, 200), 0, 0, 16, 0, 12));

        int base = Store.perMission(this), bonus = base / 2;
        LinearLayout rw = new LinearLayout(this);
        rw.setGravity(Gravity.CENTER_VERTICAL);
        rw.addView(Ui.text(this, String.format(Locale.KOREA, "+%,d원", base + bonus), 26, Ui.PRIMARY, true));
        rw.addView(Ui.text(this, String.format(Locale.KOREA, "  (기본 %,d + 보너스 %,d · 1분 기준)", base, bonus), 13, Ui.SUB, false));
        c.addView(rw, Ui.wrap());
        TextView cnt = Ui.text(this, String.format(Locale.KOREA, "누르면 다음 상황(%d/%d) 촬영 화면이 열려요", nextIdx + 1, sits.length()),
                13, Ui.SUB, false);
        cnt.setPadding(0, Ui.dp(this, 8), 0, 0);
        c.addView(cnt, Ui.wrap());
        TextView pick = Ui.text(this, "상황 골라서 찍기", 14, Ui.PRIMARY, true);
        pick.setPadding(Ui.dp(this, 12), Ui.dp(this, 12), Ui.dp(this, 12), Ui.dp(this, 2));
        pick.setBackground(Ui.pressable(android.graphics.Color.TRANSPARENT, Ui.dp(this, 12)));
        pick.setOnClickListener(v -> detail(id));
        c.addView(pick, Ui.wrap());
        return c;
    }

    private void record(String missionId, String situation) {
        startActivity(new Intent(this, RecordActivity.class).putExtra("mission", missionId).putExtra("situation", situation));
    }

    private void detail(String missionId) {
        startActivity(new Intent(this, MissionActivity.class).putExtra("mission", missionId));
    }

    // ---------- 옆으로 넘기는 미션 카드 ----------

    private int pageWidth() {
        return getResources().getDisplayMetrics().widthPixels - Ui.dp(this, 32) - Ui.dp(this, 28);   // 다음 장이 조금 보이게
    }

    private View pager() throws Exception {
        pager = new HorizontalScrollView(this) {
            @Override
            public boolean onTouchEvent(MotionEvent e) {
                boolean r = super.onTouchEvent(e);
                if (e.getAction() == MotionEvent.ACTION_UP || e.getAction() == MotionEvent.ACTION_CANCEL) post(HomeActivity.this::snap);
                return r;
            }

            @Override
            public void fling(int vx) {   // 휙 넘기면 그 방향으로 한 장
                int step = pageWidth() + Ui.dp(HomeActivity.this, 12);
                int cur = Math.round((float) getScrollX() / step);
                int to = Math.max(0, Math.min(pages.getChildCount() - 1, cur + (vx > 0 ? 1 : -1)));
                smoothScrollTo(to * step, 0);
                showDots(to);
            }
        };
        pager.setHorizontalScrollBarEnabled(false);
        pager.setClipToPadding(false);
        pages = new LinearLayout(this);
        JSONArray ms = Store.missions(this);
        for (int i = 0; i < ms.length(); i++) {
            JSONObject m = ms.getJSONObject(i);
            pages.addView(missionCard(m), Ui.margins(this, pageWidth(), -2, 0, 0, 0, 12, 0));
        }
        pages.addView(summaryCard(), Ui.margins(this, pageWidth(), -1, 0, 0, 0, 12, 0));
        pager.addView(pages);
        return pager;
    }

    private void snap() {
        int step = pageWidth() + Ui.dp(this, 12);
        int to = Math.round((float) pager.getScrollX() / step);
        pager.smoothScrollTo(to * step, 0);
        showDots(to);
    }

    private void showDots(int sel) {
        if (dots == null || pages == null) return;
        dots.removeAllViews();
        for (int i = 0; i < pages.getChildCount(); i++) {
            View d = new View(this);
            d.setBackground(Ui.round(i == sel ? Ui.PRIMARY : Ui.LINE, Ui.dp(this, 4)));
            dots.addView(d, Ui.margins(this, Ui.dp(this, i == sel ? 18 : 7), Ui.dp(this, 7), 0, 3, 0, 3, 0));
        }
    }

    /** 마지막 장: 오늘 요약 (녹화 · 적립 예정 · 동의 인원). */
    private LinearLayout summaryCard() {
        int rec = 0, expect = 0;
        String todayPrefix = Clock.sessionId().substring(0, 8);
        for (File d : Store.sessions(this)) {
            if (!d.getName().startsWith(todayPrefix) || !new File(d, "meta.json").exists()) continue;
            rec++;
            JSONObject st = Uploader.localState(d);
            String s = st.optString("state");
            expect += s.equals("done") ? st.optInt("reward") + st.optInt("appeal_extra")
                    : s.equals("rejected") || s.equals("failed") ? 0 : Store.perMission(this);
        }
        LinearLayout c = Ui.card(this);
        FrameLayout tile = Ui.iconTile(this, Icon.SPARK, Ui.ACCENT_INK, Ui.ACCENT_SOFT, 46);
        c.addView(tile, new LinearLayout.LayoutParams(Ui.dp(this, 46), Ui.dp(this, 46)));
        TextView t = Ui.text(this, "오늘 요약", 21, Ui.INK, true);
        t.setPadding(0, Ui.dp(this, 14), 0, Ui.dp(this, 10));
        c.addView(t);
        String[][] rows = {{"오늘 녹화", rec + "건"}, {"적립 예정", String.format(Locale.KOREA, "%,d원", expect)},
                {"동의 인원", Store.ownerFace(this) ? "본인 얼굴 등록됨" : "본인 얼굴 등록 필요"}};
        for (String[] r : rows) {
            LinearLayout line = new LinearLayout(this);
            line.setPadding(0, Ui.dp(this, 5), 0, Ui.dp(this, 5));
            line.addView(Ui.text(this, r[0], 14, Ui.SUB, false), Ui.weight());
            line.addView(Ui.text(this, r[1], 15, Ui.INK, true));
            c.addView(line);
        }
        TextView back = Ui.text(this, "← 앞으로 넘기면 기업이 의뢰한 미션들이 있어요", 12, Ui.SUB, false);
        back.setPadding(0, Ui.dp(this, 14), 0, 0);
        c.addView(back);
        return c;
    }

    private LinearLayout missionCard(JSONObject m) throws Exception {
        String id = m.getString("id");
        int total = m.getJSONArray("situations").length();
        int done = Store.doneSituations(this, id).size();
        LinearLayout c = Ui.card(this);
        LinearLayout top = new LinearLayout(this);
        top.setGravity(Gravity.CENTER_VERTICAL);
        FrameLayout tile = Ui.iconTile(this, Icon.TARGET, Ui.PRIMARY, Ui.PRIMARY_SOFT, 46);
        top.addView(tile, new LinearLayout.LayoutParams(Ui.dp(this, 46), Ui.dp(this, 46)));
        top.addView(new View(this), Ui.weight());
        int per = Store.perMission(this);
        boolean today = id.equals(todayId);
        Integer sc = scarce.get(id);
        int pct = (today ? 50 : 0) + (sc != null ? sc : 0);
        String label = String.format(Locale.KOREA, "+%,d원", per * (100 + pct) / 100)
                + (today ? " · 오늘 보너스" : sc != null ? " · 부족 +" + sc + "%" : "");
        top.addView(Ui.chip(this, label, Ui.ACCENT_INK, Ui.ACCENT_SOFT));
        c.addView(top);
        TextView t = Ui.text(this, m.getString("title"), 21, Ui.INK, true);
        t.setPadding(0, Ui.dp(this, 14), 0, Ui.dp(this, 6));
        c.addView(t);
        String sponsor = m.optString("sponsor");
        TextView p = Ui.text(this, (sponsor.isEmpty() ? "의뢰 · " : "의뢰 · " + sponsor + " · ") + m.getString("purpose"), 13, Ui.SUB, false);
        p.setMaxLines(2);
        p.setMinLines(2);
        c.addView(p);
        TextView n = Ui.text(this, "상황 " + done + "/" + total, 13, Ui.INK, true);
        n.setPadding(0, Ui.dp(this, 14), 0, Ui.dp(this, 8));
        c.addView(n);
        c.addView(Ui.progress(this, total == 0 ? 0f : (float) done / total), new LinearLayout.LayoutParams(-1, Ui.dp(this, 6)));
        c.addView(Ui.button(this, "상황 골라서 찍기", true, v -> detail(id)), Ui.margins(this, -1, -2, 0, 0, 16, 0, 0));
        return c;
    }
}
