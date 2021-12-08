package com.ankit.bert.tokenizerimpl;

import com.ankit.bert.tokenizer.Tokenizer;
import com.ankit.bert.utils.TokenizerUtils;
import lombok.extern.log4j.Log4j2;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Constructs a BERT tokenizer. Based on WordPiece.
 * <p>
 * This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which
 * contains most of the methods. Users should refer to the superclass for more
 * information regarding methods.
 * <p>
 * Args:
 * <p>
 * vocabFile (:obj:`string`): File containing the vocabulary.
 * <p>
 * doLowerCase (:obj:`bool`, `optional`, defaults to :obj:`True`): Whether to
 * lowercase the input when tokenizing.
 * <p>
 * doBasicTokenize (:obj:`bool`, `optional`, defaults to :obj:`True`): Whether
 * to do basic tokenization before WordPiece.
 * <p>
 * neverSplit (:obj:`bool`, `optional`, defaults to :obj:`True`): List of
 * tokens which will never be split during tokenization. Only has an effect when
 * :obj:`doBasicTokenize=True`
 * <p>
 * unkToken (:obj:`string`, `optional`, defaults to "[UNK]"): The unknown
 * token. A token that is not in the vocabulary cannot be converted to an ID and
 * is set to be this token instead.
 * <p>
 * sepToken (:obj:`string`, `optional`, defaults to "[SEP]"): The separator
 * token, which is used when building a sequence from multiple sequences, e.g.
 * two sequences for sequence classification or for a text and a question for
 * question answering. It is also used as the last token of a sequence built
 * with special tokens.
 * <p>
 * padToken (:obj:`string`, `optional`, defaults to "[PAD]"): The token used
 * for padding, for example when batching sequences of different lengths.
 * <p>
 * clsToken (:obj:`string`, `optional`, defaults to "[CLS]"): The classifier
 * token which is used when doing sequence classification (classification of the
 * whole sequence instead of per-token classification). It is the first token of
 * the sequence when built with special tokens.
 * <p>
 * maskToken (:obj:`string`, `optional`, defaults to "[MASK]"): The token used
 * for masking values. This is the token used when training this model with
 * masked language modeling. This is the token which the model will try to
 * predict.
 * <p>
 * tokenizeChineseChars (:obj:`bool`, `optional`, defaults to :obj:`True`):
 * Whether to tokenize Chinese characters. This should likely be deactivated for
 * Japanese: see: https://github.com/huggingface/transformers/issues/328
 */

@Log4j2
public class BertTokenizer implements Tokenizer {

    private final String vocabFile = "vocab.txt";
    private Map<String, Integer> tokenIdMap;
    private Map<Integer, String> idTokenMap;
    private final boolean doLowerCase = true;
    private final boolean doBasicTokenize = true;
    private final List<String> neverSplit = new ArrayList<>();
    private final String unkToken = "[UNK]";
    private final String sepToken = "[SEP]";
    private final String padToken = "[PAD]";
    private final String clsToken = "[CLS]";
    private final String maskToken = "[MASK]";
    private final boolean tokenizeChineseChars = true;
    private BasicTokenizer basicTokenizer;
    private WordPieceTokenizer wordPieceTokenizer;

    private static final int MAXLEN = 512;


    public BertTokenizer() {
        this.init();
    }

    private void init() {
        try {
            this.tokenIdMap = loadVocab(vocabFile);
        } catch (IOException e) {
            log.error("Unable to load vocab due to: ", e);
        }
        this.idTokenMap = new HashMap<>();
        for (String key : tokenIdMap.keySet()) {
            this.idTokenMap.put(tokenIdMap.get(key), key);
        }

        if (doBasicTokenize) {
            this.basicTokenizer = new BasicTokenizer(doLowerCase, tokenizeChineseChars);
        }
        this.wordPieceTokenizer = new WordPieceTokenizer(tokenIdMap, unkToken);
    }

    private Map<String, Integer> loadVocab(String vocabFileName) throws IOException {
        ClassLoader classloader = Thread.currentThread().getContextClassLoader();
        InputStream file = classloader.getResourceAsStream(vocabFileName);
        return TokenizerUtils.generateTokenIdMap(file);
    }

    /**
     * Tokenizes a piece of text into its word pieces.
     * <p>
     * This uses a greedy longest-match-first algorithm to perform tokenization
     * using the given vocabulary.
     * <p>
     * For example: input = "unaffable" output = ["un", "##aff", "##able"]
     * <p>
     * Args: text: A single token or whitespace separated tokens. This should have
     * already been passed through `BasicTokenizer`.
     * <p>
     * Returns: A list of wordpiece tokens.
     */
    @Override
    public List<String> tokenize(String text) {
        List<String> splitTokens = new ArrayList<>();
        if (doBasicTokenize) {
            for (String token : basicTokenizer.tokenize(text)) {
                splitTokens.addAll(wordPieceTokenizer.tokenize(token));
            }
        } else {
            splitTokens = wordPieceTokenizer.tokenize(text);
        }
        return splitTokens;
    }

    public String convertTokensToString(List<String> tokens) {
        // Converts a sequence of tokens (string) in a single string.
        return tokens.stream().map(s -> s.replace("##", "")).collect(Collectors.joining(" "));
    }

    public List<Integer> convertTokensToIds(List<String> tokens) {
        List<Integer> output = new ArrayList<>();
        for (String s : tokens) {
            output.add(tokenIdMap.get(s));
        }
        return output;
    }

    public int vocabSize() {
        return tokenIdMap.size();
    }

    public List<Integer> convertTokensToMasks(List<Integer> tokenize) {
        List<Integer> resultMask = new ArrayList<>(tokenize);
        for (int i = 0, resultMaskSize = resultMask.size(); i < resultMaskSize; i++) {
            Integer token = resultMask.get(i);
            if (token == 0) {
                return resultMask;
            }
            resultMask.set(i, 1);
        }
        return resultMask;
    }
}
