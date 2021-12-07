package com.ankit.bert;

import com.ankit.bert.tokenizerimpl.BertTokenizer;

import java.util.List;

public class Main {
    public static void main(String[] args) {
        BertTokenizer bertTokenizer = new BertTokenizer();
        String text = "所有的数据在存储和运算时都要使用二进制数表示";
        List<String> tokenize = bertTokenizer.tokenize(text);
        List<Integer> tokensToIds = bertTokenizer.convertTokensToIds(tokenize);
        System.out.println(tokenize);
        System.out.println(tokensToIds);
    }
}
