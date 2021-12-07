package com.ankit.bert.tokenizerimpl;

import com.ankit.bert.tokenizer.Tokenizer;
import com.ankit.bert.utils.TokenizerUtils;

import java.util.ArrayList;
import java.util.List;

public class BasicTokenizer implements Tokenizer {
    private final boolean doLowerCase;
    private final boolean tokenizeChineseChars;
    private final List<String> neverSplit = new ArrayList<>();

    public BasicTokenizer(boolean doLowerCase, boolean tokenizeChineseChars) {
        this.doLowerCase = doLowerCase;
        this.tokenizeChineseChars = tokenizeChineseChars;
    }

    @Override
    public List<String> tokenize(String text) {
        text = TokenizerUtils.cleanText(text);
        if (tokenizeChineseChars) {
            text = TokenizerUtils.tokenizeChineseChars(text);
        }
        List<String> origTokens = TokenizerUtils.whitespaceTokenize(text);

        List<String> splitTokens = new ArrayList<>();
        for (String token : origTokens) {
            if (doLowerCase && !neverSplit.contains(token)) {
                token = TokenizerUtils.runStripAccents(token.toLowerCase());
                splitTokens.addAll(TokenizerUtils.runSplitOnPunc(token, neverSplit));
            }
        }
        return TokenizerUtils.whitespaceTokenize(String.join(" ", splitTokens));
    }

}
