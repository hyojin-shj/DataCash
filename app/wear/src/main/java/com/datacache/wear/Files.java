package com.datacache.wear;

import java.io.File;
import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;

final class Files {
    private Files() {}

    static void write(File f, String s) throws Exception {
        try (FileOutputStream o = new FileOutputStream(f)) {
            o.write(s.getBytes(StandardCharsets.UTF_8));
        }
    }
}
