package jpountz;

import java.util.Arrays;
import java.util.concurrent.TimeUnit;
import java.util.function.BiConsumer;

import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorShuffle;

@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Fork(value = 1, jvmArgsPrepend = {"--add-modules=jdk.incubator.vector"})
@State(Scope.Benchmark)
@BenchmarkMode(Mode.Throughput)
public class PrefixSumBenchmark {

  // See this good resource on using SIMD for prefix sums: https://en.algorithmica.org/hpc/algorithms/prefix/

  private static final VectorShuffle<Integer> IOTA1_256 = VectorShuffle.iota(IntVector.SPECIES_256, -1, 1, true);
  private static final VectorMask<Integer> MASK1_256 = VectorMask.fromValues(IntVector.SPECIES_256, false, true, true, true, true, true, true, true);

  @Benchmark
  public void prefixSumVector256_v2(PrefixSumState state, Blackhole bh) {
    int[] input = state.input;
    int[] output = state.output;

    IntVector vec0 = IntVector.fromArray(IntVector.SPECIES_256, input, 0);
//    vec0 = vec0.add(vec0);
    vec0 = vec0.rearrange(IOTA1_256);
    vec0.intoArray(output, 0);

    bh.consume(output);
  }

}
