package com.ankit.bert.tokenizerimpl;

import com.ankit.bert.tokenizer.Tokenizer;
import com.ankit.bert.utils.TokenizerUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class WordPieceTokenizer implements Tokenizer {
    private final Map<String, Integer> vocab;
    private final String unkToken;
    private final int maxInputCharsPerWord;

    public WordPieceTokenizer(Map<String, Integer> vocab, String unkToken, int maxInputCharsPerWord) {
        this.vocab = vocab;
        this.unkToken = unkToken;
        this.maxInputCharsPerWord = maxInputCharsPerWord;
    }

    public WordPieceTokenizer(Map<String, Integer> vocab, String unkToken) {
        this(vocab, unkToken, 100);
    }

    @Override
    public List<String> tokenize(String text) {
        /*
          Tokenizes a piece of text into its word pieces.

          This uses a greedy longest-match-first algorithm to perform tokenization
          using the given vocabulary.

          For example: input = "unaffable" output = ["un", "##aff", "##able"]

          Args: text: A single token or whitespace separated tokens. This should have
          already been passed through `BasicTokenizer`.

          Returns: A list of word piece tokens.

         */

        List<String> outputTokens = new ArrayList<>();
        for (String token : TokenizerUtils.whitespaceTokenize(text)) {
            if (token.length() > maxInputCharsPerWord) {
                outputTokens.add(unkToken);
                continue;
            }
            boolean isBad = false;
            int start = 0;

            List<String> subTokens = new ArrayList<>();
            while (start < token.length()) {
                int end = token.length();
                String curSubStr = "";
                while (start < end) {
                    String subStr = token.substring(start, end);
                    if (start > 0) {
                        subStr = "##" + subStr;
                    }
                    if (vocab.containsKey(subStr)) {
                        curSubStr = subStr;
                        break;
                    }
                    end--;
                }
                if (curSubStr.isEmpty()) {
                    isBad = true;
                    break;
                }
                subTokens.add(curSubStr);
                start = end;
            }
            if (isBad) {
                outputTokens.add(unkToken);
            } else {
                outputTokens.addAll(subTokens);
            }
        }
        return outputTokens;
    }
}
