package edu.uw;

import java.util.Arrays;
import java.util.Iterator;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;
import java.util.List;
import java.util.function.Supplier;
import java.io.File;

import com.google.protobuf.InvalidProtocolBufferException;

import edu.uw.TaggerflowProtos.SparseValue;
import edu.uw.TaggerflowProtos.TaggedSentence;
import edu.uw.TaggerflowProtos.TaggedToken;
import edu.uw.TaggerflowProtos.TaggingInput;
import edu.uw.TaggerflowProtos.TaggingResult;
import edu.uw.TaggerflowProtos.TaggerInitialization;

public class Taggerflow {
    private final long session;
    public Taggerflow(File modelPath, double beam) {
        System.loadLibrary("taggerflow");
        session = initialize(TaggerInitialization.newBuilder()
                             .setModelPath(modelPath.getAbsolutePath())
                             .setBeam(beam)
                             .build()
                             .toByteArray());
    }

    public Stream<TaggedSentence> predict(String filename, int maxBatchSize) {
        queueFile(filename, session);
        return predictionStream(() -> {
                try {
                    return TaggingResult.parseFrom(predictRemainingFromFile(maxBatchSize, session));
                } catch (final InvalidProtocolBufferException e) {
                    throw new RuntimeException(e);
                }
            });
    }

    public Stream<TaggedSentence> predict(TaggingInput input) {
        queueInput(input.toByteArray(), session);
        return predictionStream(() -> {
                try {
                    return TaggingResult.parseFrom(predictRemainingFromInput(session));
                } catch (final InvalidProtocolBufferException e) {
                    throw new RuntimeException(e);
                }
            });
    }

    public Stream<TaggedSentence> predictionStream(Supplier<TaggingResult> resultSupplier) {
        final TaggingResult firstResult = resultSupplier.get();
        final Iterable<TaggedSentence> sentenceIterable = () -> new Iterator<TaggedSentence>() {
            TaggingResult result = firstResult;
            int i = 0;

            @Override
            public boolean hasNext() {
                return i < result.getSentenceCount() || result.getHasMore();
            }

            @Override
            public TaggedSentence next() {
                if (i >= result.getSentenceCount()) {
                    result = resultSupplier.get();
                    i = 0;
                }
                return result.getSentence(i++);
            }
        };
        return StreamSupport.stream(sentenceIterable.spliterator(), false);
    }

    @Override
    protected void finalize() throws Throwable {
        close(session);
        super.finalize();
    }

    private static native void close(long session);

    private static native long initialize(byte[] packed);

    private static native void queueInput(byte[] packedInput, long session);
    private static native byte[] predictRemainingFromInput(long session);

    private static native void queueFile(String filename, long session);
    private static native byte[] predictRemainingFromFile(int maxBatchSize, long session);

    public static void main(String[] args) {
        if (args.length != 1) {
            System.err.println("Missing model directory.");
            System.exit(1);
        }
        final File modelPath = new File(args[0]);
        final Taggerflow tagger = new Taggerflow(modelPath, 1e-4f);

        final TaggingInput.Builder builder = TaggingInput.newBuilder();
        builder.addSentenceBuilder().addAllWord(Arrays.asList("Visiting relatives can be boring .".split(" ")));

        final TaggedSentence sentence = tagger.predict(builder.build()).findFirst().get();
        for (final TaggedToken token : sentence.getTokenList()) {
            System.out.print(token.getWord());
            for (final SparseValue score : token.getScoreList()) {
                System.out.print(String.format("|%d=%.4f", score.getIndex(), score.getValue()));
            }
            System.out.println();
        }
    }
}
