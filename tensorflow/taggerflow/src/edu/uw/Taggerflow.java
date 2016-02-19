package edu.uw;

import java.util.Arrays;
import java.util.List;

import com.google.protobuf.InvalidProtocolBufferException;

import edu.uw.TaggerflowProtos.SparseValue;
import edu.uw.TaggerflowProtos.TaggedSentence;
import edu.uw.TaggerflowProtos.TaggedToken;
import edu.uw.TaggerflowProtos.TaggingInput;
import edu.uw.TaggerflowProtos.TaggingResult;

public class Taggerflow {
    private final long session;
    public Taggerflow(String model, String spacesDir) {
        System.loadLibrary("taggerflow");
        session = initialize(model, spacesDir);
    }

    public TaggingResult predict(String filename, int maxBatchSize) {
        try {
            return TaggingResult.parseFrom(predictPacked(filename, maxBatchSize, session));
        } catch (final InvalidProtocolBufferException e) {
            throw new RuntimeException(e);
        }
    }

    public TaggingResult predict(TaggingInput input) {
        try {
            return TaggingResult.parseFrom(predictPacked(input.toByteArray(), session));
        } catch (final InvalidProtocolBufferException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    protected void finalize() throws Throwable {
        close(session);
        super.finalize();
    }

    private static native void close(long session);

    private static native long initialize(String model, String spacesDir);

    private static native byte[] predictPacked(byte[] packed, long session);

    private static native byte[] predictPacked(String filename, int maxBatchSize, long session);


    public static void main(String[] args) {
        if (args.length != 1) {
            System.err.println("Missing model directory.");
            System.exit(1);
        }
        final String modelDir = args[0];
        final Taggerflow tagger = new Taggerflow(modelDir + "/graph.pb", modelDir);

        final TaggingInput.Builder builder = TaggingInput.newBuilder();
        builder.addSentenceBuilder().addAllWord(Arrays.asList("Visiting relatives can be boring .".split(" ")));
        final TaggingResult result = tagger.predict(builder.build());

        final TaggedSentence sentence = result.getSentence(0);
        for (final TaggedToken token : sentence.getTokenList()) {
            System.out.print(token.getWord());
            for (final SparseValue score : token.getScoreList()) {
                System.out.print(String.format("|%d=%.4f", score.getIndex(), score.getValue()));
            }
            System.out.println();
        }
    }
}
