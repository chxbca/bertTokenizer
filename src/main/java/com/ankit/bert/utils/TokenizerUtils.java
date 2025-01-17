package com.ankit.bert.utils;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.text.Normalizer;
import java.text.Normalizer.Form;
import java.util.*;

public class TokenizerUtils {

    private static final String WHITE_SPACE = " ";

    public static String cleanText(String text) {
        // Performs invalid character removal and whitespace cleanup on text."""
        StringBuilder output = new StringBuilder();
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (c == 0 || c == 0xFFFD || isControl(c)) {
                continue;
            }
            output.append(isWhitespace(c) ? WHITE_SPACE : c);
        }
        return output.toString();
    }

    public static String tokenizeChineseChars(String text) {
        // Adds whitespace around any CJK character.
        StringBuilder output = new StringBuilder();
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (isChineseChar(c)) {
                output.append(WHITE_SPACE).append(c).append(WHITE_SPACE);
            } else {
                output.append(c);
            }
        }
        return output.toString();
    }

    public static List<String> whitespaceTokenize(String text) {
        // Runs basic whitespace cleaning and splitting on a piece of text.
        text = text.trim();
        if (!text.isEmpty()) {
            return Arrays.asList(text.split("\\s+"));
        }
        return new ArrayList<>();

    }

    public static String runStripAccents(String token) {
        token = Normalizer.normalize(token, Form.NFD);
        StringBuilder output = new StringBuilder();
        for (int i = 0; i < token.length(); i++) {
            char c = token.charAt(i);
            if (Character.NON_SPACING_MARK != Character.getType(c)) {
                output.append(c);
            }
        }
        return output.toString();
    }

    public static List<String> runSplitOnPunc(String token, List<String> neverSplit) {
        // Splits punctuation on a piece of text.
        List<String> output = new ArrayList<>();
        if (neverSplit != null && neverSplit.contains(token)) {
            output.add(token);
            return output;
        }

        boolean startNewWord = true;
        StringBuilder str = new StringBuilder();
        for (int i = 0; i < token.length(); i++) {
            Character c = token.charAt(i);
            if (isPunctuation(c)) {
                if (str.length() > 0) {
                    output.add(str.toString());
                    str.setLength(0);
                }
                output.add(c.toString());
                startNewWord = true;
            } else {
                if (startNewWord && str.length() > 0) {
                    output.add(str.toString());
                    str.setLength(0);
                }
                startNewWord = false;
                str.append(c);
            }
        }
        if (str.length() > 0) {
            output.add(str.toString());
        }
        return output;
    }

    public static Map<String, Integer> generateTokenIdMap(InputStream file) throws IOException {
        HashMap<String, Integer> tokenIdMap = new HashMap<>();
        if (file == null) {
            return tokenIdMap;
        }

        try (InputStreamReader inputStreamReader = new InputStreamReader(file);
             BufferedReader br = new BufferedReader(inputStreamReader)) {
            String line;
            int index = 0;
            while ((line = br.readLine()) != null) {
                tokenIdMap.put(line, index);
                index++;
            }
        }
        return tokenIdMap;
    }

    private static boolean isPunctuation(char c) {
        // Checks whether `chars` is a punctuation character.
        // We treat all non-letter/number ASCII as punctuation.
        // Characters such as "^", "$", and "`" are not in the Unicode
        // Punctuation class but we treat them as punctuation anyways, for
        // consistency.
        boolean isPunctuation = (c >= 33 && c <= 47) ||
                (c >= 58 && c <= 64) ||
                (c >= 91 && c <= 96) ||
                (c >= 123 && c <= 126);
        if (isPunctuation) {
            return true;
        }
        int charType = Character.getType(c);
        return Character.CONNECTOR_PUNCTUATION == charType ||
                Character.DASH_PUNCTUATION == charType ||
                Character.END_PUNCTUATION == charType ||
                Character.FINAL_QUOTE_PUNCTUATION == charType ||
                Character.INITIAL_QUOTE_PUNCTUATION == charType ||
                Character.OTHER_PUNCTUATION == charType ||
                Character.START_PUNCTUATION == charType;
    }

    private static boolean isWhitespace(char c) {
        // Checks whether `chars` is a whitespace character.
        // \t, \n, and \r are technically contorl characters but we treat them
        // as whitespace since they are generally considered as such.
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            return true;
        }

        int charType = Character.getType(c);
        return Character.SPACE_SEPARATOR == charType;
    }

    private static boolean isControl(char c) {
        // Checks whether `chars` is a control character.
        // These are technically control characters but we count them as whitespace
        // characters.
        if (c == '\t' || c == '\n' || c == '\r') {
            return false;
        }

        int charType = Character.getType(c);
        return Character.CONTROL == charType ||
                Character.DIRECTIONALITY_COMMON_NUMBER_SEPARATOR == charType ||
                Character.FORMAT == charType ||
                Character.PRIVATE_USE == charType ||
                Character.SURROGATE == charType ||
                Character.UNASSIGNED == charType;
    }

    private static boolean isChineseChar(int cp) {
        // Checks whether CP is the codepoint of a CJK character."""
        // This defines a "chinese character" as anything in the CJK Unicode block:
        // https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        //
        // Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        // despite its name. The modern Korean Hangul alphabet is a different block,
        // as is Japanese Hiragana and Katakana. Those alphabets are used to write
        // space-separated words, so they are not treated specially and handled
        // like the all of the other languages.
        return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0x20000 && cp <= 0x2A6DF)
                || (cp >= 0x2A700 && cp <= 0x2B73F) || (cp >= 0x2B740 && cp <= 0x2B81F)
                || (cp >= 0x2B820 && cp <= 0x2CEAF) || (cp >= 0xF900 && cp <= 0xFAFF)
                || (cp >= 0x2F800 && cp <= 0x2FA1F);
    }
}
