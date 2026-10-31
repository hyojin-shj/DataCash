package com.datacash.wear;

import java.util.Locale;

/** 워치 화면 글자: 워치 언어가 한국어면 한국어, 아니면 영어 (화면이 작아 영어는 짧게). */
final class Tr {
    private Tr() {}

    static boolean ko() {
        return "ko".equals(Locale.getDefault().getLanguage());
    }

    static String s(String ko, String en) {
        return ko() ? ko : en;
    }
}
