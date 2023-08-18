package jpountz;

import java.util.Arrays;
import java.util.concurrent.TimeUnit;
import java.util.function.BiConsumer;

import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
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
public class PrefixSumBenchmark {

  // See this good resource on using SIMD for prefix sums: https://en.algorithmica.org/hpc/algorithms/prefix/

  @Setup(Level.Trial)
  public void setup() {
    sanity();
  }

  private static final VectorSpecies<Integer> SPECIES_128 = IntVector.SPECIES_128;
  private static final VectorSpecies<Long> L_SPECIES_128 = LongVector.SPECIES_128;

  private static final VectorSpecies<Integer> SPECIES_512 = IntVector.SPECIES_512;
  private static final VectorSpecies<Long> L_SPECIES_512 = LongVector.SPECIES_512;

  private static final VectorSpecies<Integer> SPECIES_FLEX = IntVector.SPECIES_PREFERRED;

  private static final int[] tmpInts = new int[128];
  private static final long[] tmpLongs = new long[64];
  private static final int INT_MASK = (1 << 4) - 1;
  private static final long LONG_MASK = ((1L << 4) - 1) | (((1L << 4) - 1) << 32);

//  @Benchmark
  public void scalarDecode_ScalarPrefixSum(PrefixSumState state, Blackhole bh) {
    int[] input = state.inputPackedInts;
    int[] output = state.output;

    scalarDecode(input, output);
    scalarPrefixSum(output);

    bh.consume(output);
  }

//  @Benchmark
  public void vectorDecode_ScalarPrefixSum(PrefixSumState state, Blackhole bh) {
    int[] input = state.inputPackedInts;
    int[] output = state.output;

    vectorDecode(input, output);
    scalarPrefixSum(output);

    bh.consume(output);
  }

//  @Benchmark
  public void vectorDecode512_ScalarPrefixSum(PrefixSumState state, Blackhole bh) {
    int[] input = state.inputPackedInts512;
    int[] output = state.output;

    vectorDecode512(input, output);
    scalarPrefixSum(output);

    bh.consume(output);
  }

//  @Benchmark
  public void scalarDecodeFlex_ScalarPrefixSum(PrefixSumState state, Blackhole bh) {
    int[] input = state.inputPackedIntsFlex;
    int[] output = state.output;

    scalarDecodeFlex(input, output);
    scalarPrefixSum(output);

    bh.consume(output);
  }

//  @Benchmark
  public void vectorDecodeFlex_ScalarPrefixSum(PrefixSumState state, Blackhole bh) {
    int[] input = state.inputPackedIntsFlex;
    int[] output = state.output;

    vectorDecodeFlex(input, output);
    scalarPrefixSum(output);

    bh.consume(output);
  }

//  @Benchmark
  public void scalarDecode_VectorPrefixSum(PrefixSumState state, Blackhole bh) {
    int[] input = state.inputPackedInts;
    int[] output = state.output;

    scalarDecode(input, tmpInts);
    vectorPrefixSum(tmpInts, output);

    bh.consume(output);
  }

//  @Benchmark
  public void vectorDecode_VectorPrefixSum_TwoPhase(PrefixSumState state, Blackhole bh) {
    int[] input = state.inputPackedInts;
    int[] output = state.output;

    vectorDecode(input, tmpInts);
    vectorPrefixSum(tmpInts, output);

    bh.consume(output);
  }

//  @Benchmark
  public void scalarDecodeTo32_ScalarPrefixSum(PrefixSumState state, Blackhole bh) {
    long[] input = state.inputPackedLongs;
    int[] output = state.output;

    scalarDecodeTo32(input, tmpLongs);
    scalarPrefixSum32(tmpLongs, output);

    bh.consume(output);
  }

//  @Benchmark
  public void vectorDecodeTo32_ScalarPrefixSum(PrefixSumState state, Blackhole bh) {
    long[] input = state.inputPackedLongs;
    int[] output = state.output;

    vectorDecodeTo32(input, tmpLongs);
    scalarPrefixSum32(tmpLongs, output);

    bh.consume(output);
  }

//  @Benchmark
  public void vectorDecodeTo32_512_ScalarPrefixSum(PrefixSumState state, Blackhole bh) {
    long[] input = state.inputPackedLongs;
    int[] output = state.output;

    vectorDecodeTo32_512(input, tmpLongs);
    scalarPrefixSum32(tmpLongs, output);

    bh.consume(output);
  }


//  @Benchmark
  public void scalarDecodeTo32_VectorPrefixSum(PrefixSumState state, Blackhole bh) {
    long[] input = state.inputPackedLongs;
    int[] output = state.output;

    scalarDecodeTo32(input, tmpLongs);
//    prefixSum32ScalarInlined(tmpLongs, output);

    bh.consume(output);
  }

//  @Benchmark
  public void vectorDecodeTo32_VectorPrefixSum(PrefixSumState state, Blackhole bh) {
    long[] input = state.inputPackedLongs;
    int[] output = state.output;

    vectorDecodeTo32(input, tmpLongs);
//    prefixSum32ScalarInlined(tmpLongs, output);

    bh.consume(output);
  }


//  @Benchmark
  public void scalarDecode(PrefixSumState state, Blackhole bh) {
    int[] input = state.inputPackedInts;
    int[] output = state.output;

    scalarDecode(input, output);

    bh.consume(output);
  }

  @Benchmark
  public void scalarDecodeFlex(PrefixSumState state, Blackhole bh) {
    int[] input = state.inputPackedIntsFlex;
    int[] output = state.output;

    scalarDecodeFlex(input, output);

    bh.consume(output);
  }

//  @Benchmark
  public void vectorDecode(PrefixSumState state, Blackhole bh) {
    int[] input = state.inputPackedInts;
    int[] output = state.output;

    vectorDecode(input, output);

    bh.consume(output);
  }

  @Benchmark
  public void vectorDecodeFlex(PrefixSumState state, Blackhole bh) {
    int[] input = state.inputPackedIntsFlex;
    int[] output = state.output;

    vectorDecodeFlex(input, output);

    bh.consume(output);
  }


  private static void scalarDecode(int[] input, int[] output) {
    int outBase = 0;
    for (int i = 0; i < 8; i++) {
      int shift = 0;
      for (int j = 0; j < 4; j++) {
        output[outBase] = (input[0] >>> shift) & INT_MASK;
        output[outBase + 1] = (input[1] >>> shift) & INT_MASK;
        output[outBase + 2] = (input[2] >>> shift) & INT_MASK;
        output[outBase + 3] = (input[3] >>> shift) & INT_MASK;
        shift += 4;
        outBase += 4;
      }
    }
  }

  private static void vectorDecode(int[] input, int[] output) {
    IntVector inVec = IntVector.fromArray(SPECIES_128, input, 0);
    IntVector outVec;
    int inOff = 0;
    int outOff = 0;

    outVec = inVec.and(INT_MASK);
    outVec.intoArray(output, outOff);

    outVec = inVec.lanewise(VectorOperators.LSHR, 4).and(INT_MASK);
    outVec.intoArray(output, outOff+=4);

    outVec = inVec.lanewise(VectorOperators.LSHR, 8).and(INT_MASK);
    outVec.intoArray(output, outOff+=4);

    outVec = inVec.lanewise(VectorOperators.LSHR, 12).and(INT_MASK);
    outVec.intoArray(output, outOff+=4);

    outVec = inVec.lanewise(VectorOperators.LSHR, 16).and(INT_MASK);
    outVec.intoArray(output, outOff+=4);

    outVec = inVec.lanewise(VectorOperators.LSHR, 20).and(INT_MASK);
    outVec.intoArray(output, outOff+=4);

    outVec = inVec.lanewise(VectorOperators.LSHR, 24).and(INT_MASK);
    outVec.intoArray(output, outOff+=4);

    outVec = inVec.lanewise(VectorOperators.LSHR, 28).and(INT_MASK);
    outVec.intoArray(output, outOff+=4);

    inVec = IntVector.fromArray(SPECIES_128, input, inOff+=4);

    outVec = inVec.and(INT_MASK);
    outVec.intoArray(output, outOff+=4);

    outVec = inVec.lanewise(VectorOperators.LSHR, 4).and(INT_MASK);
    outVec.intoArray(output, outOff+=4);

    outVec = inVec.lanewise(VectorOperators.LSHR, 8).and(INT_MASK);
    outVec.intoArray(output, outOff+=4);

    outVec = inVec.lanewise(VectorOperators.LSHR, 12).and(INT_MASK);
    outVec.intoArray(output, outOff+=4);

    outVec = inVec.lanewise(VectorOperators.LSHR, 16).and(INT_MASK);
    outVec.intoArray(output, outOff+=4);

    outVec = inVec.lanewise(VectorOperators.LSHR, 20).and(INT_MASK);
    outVec.intoArray(output, outOff+=4);

    outVec = inVec.lanewise(VectorOperators.LSHR, 24).and(INT_MASK);
    outVec.intoArray(output, outOff+=4);

    outVec = inVec.lanewise(VectorOperators.LSHR, 28).and(INT_MASK);
    outVec.intoArray(output, outOff+=4);

    inVec = IntVector.fromArray(SPECIES_128, input, inOff+=4);

    outVec = inVec.and(INT_MASK);
    outVec.intoArray(output, outOff+=4);

    outVec = inVec.lanewise(VectorOperators.LSHR, 4).and(INT_MASK);
    outVec.intoArray(output, outOff+=4);

    outVec = inVec.lanewise(VectorOperators.LSHR, 8).and(INT_MASK);
    outVec.intoArray(output, outOff+=4);

    outVec = inVec.lanewise(VectorOperators.LSHR, 12).and(INT_MASK);
    outVec.intoArray(output, outOff+=4);

    outVec = inVec.lanewise(VectorOperators.LSHR, 16).and(INT_MASK);
    outVec.intoArray(output, outOff+=4);

    outVec = inVec.lanewise(VectorOperators.LSHR, 20).and(INT_MASK);
    outVec.intoArray(output, outOff+=4);

    outVec = inVec.lanewise(VectorOperators.LSHR, 24).and(INT_MASK);
    outVec.intoArray(output, outOff+=4);

    outVec = inVec.lanewise(VectorOperators.LSHR, 28).and(INT_MASK);
    outVec.intoArray(output, outOff+=4);

    inVec = IntVector.fromArray(SPECIES_128, input, inOff+=4);

    outVec = inVec.and(INT_MASK);
    outVec.intoArray(output, outOff+=4);

    outVec = inVec.lanewise(VectorOperators.LSHR, 4).and(INT_MASK);
    outVec.intoArray(output, outOff+=4);

    outVec = inVec.lanewise(VectorOperators.LSHR, 8).and(INT_MASK);
    outVec.intoArray(output, outOff+=4);

    outVec = inVec.lanewise(VectorOperators.LSHR, 12).and(INT_MASK);
    outVec.intoArray(output, outOff+=4);

    outVec = inVec.lanewise(VectorOperators.LSHR, 16).and(INT_MASK);
    outVec.intoArray(output, outOff+=4);

    outVec = inVec.lanewise(VectorOperators.LSHR, 20).and(INT_MASK);
    outVec.intoArray(output, outOff+=4);

    outVec = inVec.lanewise(VectorOperators.LSHR, 24).and(INT_MASK);
    outVec.intoArray(output, outOff+=4);

    outVec = inVec.lanewise(VectorOperators.LSHR, 28).and(INT_MASK);
    outVec.intoArray(output, outOff+=4);
  }

  private static void vectorDecode512(int[] input, int[] output) {
    IntVector inVec = IntVector.fromArray(SPECIES_512, input, 0);

    IntVector outVec = inVec.and(INT_MASK);
    outVec.intoArray(output, 0);

    outVec = inVec.lanewise(VectorOperators.LSHR, 4).and(INT_MASK);
    outVec.intoArray(output, 16);

    outVec = inVec.lanewise(VectorOperators.LSHR, 8).and(INT_MASK);
    outVec.intoArray(output, 32);

    outVec = inVec.lanewise(VectorOperators.LSHR, 12).and(INT_MASK);
    outVec.intoArray(output, 48);

    outVec = inVec.lanewise(VectorOperators.LSHR, 16).and(INT_MASK);
    outVec.intoArray(output, 64);

    outVec = inVec.lanewise(VectorOperators.LSHR, 20).and(INT_MASK);
    outVec.intoArray(output, 80);

    outVec = inVec.lanewise(VectorOperators.LSHR, 24).and(INT_MASK);
    outVec.intoArray(output, 96);

    outVec = inVec.lanewise(VectorOperators.LSHR, 28).and(INT_MASK);
    outVec.intoArray(output, 112);
  }

  private static void scalarDecodeFlex(int[] input, int[] output) {
    int outBase = 0;
    int shift = 0;
    for (int i = 0; i < 8; i++) {
      output[outBase] = (input[0] >>> shift) & INT_MASK;
      output[outBase + 1] = (input[1] >>> shift) & INT_MASK;
      output[outBase + 2] = (input[2] >>> shift) & INT_MASK;
      output[outBase + 3] = (input[3] >>> shift) & INT_MASK;
      output[outBase + 4] = (input[4] >>> shift) & INT_MASK;
      output[outBase + 5] = (input[5] >>> shift) & INT_MASK;
      output[outBase + 6] = (input[6] >>> shift) & INT_MASK;
      output[outBase + 7] = (input[7] >>> shift) & INT_MASK;
      output[outBase + 8] = (input[8] >>> shift) & INT_MASK;
      output[outBase + 9] = (input[9] >>> shift) & INT_MASK;
      output[outBase + 10] = (input[10] >>> shift) & INT_MASK;
      output[outBase + 11] = (input[11] >>> shift) & INT_MASK;
      output[outBase + 12] = (input[12] >>> shift) & INT_MASK;
      output[outBase + 13] = (input[13] >>> shift) & INT_MASK;
      output[outBase + 14] = (input[14] >>> shift) & INT_MASK;
      output[outBase + 15] = (input[15] >>> shift) & INT_MASK;
      shift += 4;
      outBase += 16;
    }
  }

  private static void vectorDecodeFlex(int[] input, int[] output) {
    IntVector inVec;
    IntVector outVec;

    int inOff = 0;
    int outOff;

    int lanes = SPECIES_FLEX.length();
    assert 16 % lanes == 0;
    int bound = 16 / lanes;
    for (int i = 0; i < bound; i++) {
      outOff = i * lanes;

      inVec = IntVector.fromArray(SPECIES_FLEX, input, inOff);

      outVec = inVec.and(INT_MASK);
      outVec.intoArray(output, outOff);

      outVec = inVec.lanewise(VectorOperators.LSHR, 4).and(INT_MASK);
      outVec.intoArray(output, outOff + 16);

      outVec = inVec.lanewise(VectorOperators.LSHR, 8).and(INT_MASK);
      outVec.intoArray(output, outOff + 32);

      outVec = inVec.lanewise(VectorOperators.LSHR, 12).and(INT_MASK);
      outVec.intoArray(output, outOff + 48);

      outVec = inVec.lanewise(VectorOperators.LSHR, 16).and(INT_MASK);
      outVec.intoArray(output, outOff + 64);

      outVec = inVec.lanewise(VectorOperators.LSHR, 20).and(INT_MASK);
      outVec.intoArray(output, outOff + 80);

      outVec = inVec.lanewise(VectorOperators.LSHR, 24).and(INT_MASK);
      outVec.intoArray(output, outOff + 96);

      outVec = inVec.lanewise(VectorOperators.LSHR, 28).and(INT_MASK);
      outVec.intoArray(output, outOff + 112);

      inOff += lanes;
    }
  }

  private static void scalarPrefixSum(int[] input) {
    input[1] += input[0];
    input[2] += input[1];
    input[3] += input[2];
    input[4] += input[3];
    input[5] += input[4];
    input[6] += input[5];
    input[7] += input[6];
    input[8] += input[7];
    input[9] += input[8];
    input[10] += input[9];
    input[11] += input[10];
    input[12] += input[11];
    input[13] += input[12];
    input[14] += input[13];
    input[15] += input[14];
    input[16] += input[15];
    input[17] += input[16];
    input[18] += input[17];
    input[19] += input[18];
    input[20] += input[19];
    input[21] += input[20];
    input[22] += input[21];
    input[23] += input[22];
    input[24] += input[23];
    input[25] += input[24];
    input[26] += input[25];
    input[27] += input[26];
    input[28] += input[27];
    input[29] += input[28];
    input[30] += input[29];
    input[31] += input[30];
    input[32] += input[31];
    input[33] += input[32];
    input[34] += input[33];
    input[35] += input[34];
    input[36] += input[35];
    input[37] += input[36];
    input[38] += input[37];
    input[39] += input[38];
    input[40] += input[39];
    input[41] += input[40];
    input[42] += input[41];
    input[43] += input[42];
    input[44] += input[43];
    input[45] += input[44];
    input[46] += input[45];
    input[47] += input[46];
    input[48] += input[47];
    input[49] += input[48];
    input[50] += input[49];
    input[51] += input[50];
    input[52] += input[51];
    input[53] += input[52];
    input[54] += input[53];
    input[55] += input[54];
    input[56] += input[55];
    input[57] += input[56];
    input[58] += input[57];
    input[59] += input[58];
    input[60] += input[59];
    input[61] += input[60];
    input[62] += input[61];
    input[63] += input[62];
    input[64] += input[63];
    input[65] += input[64];
    input[66] += input[65];
    input[67] += input[66];
    input[68] += input[67];
    input[69] += input[68];
    input[70] += input[69];
    input[71] += input[70];
    input[72] += input[71];
    input[73] += input[72];
    input[74] += input[73];
    input[75] += input[74];
    input[76] += input[75];
    input[77] += input[76];
    input[78] += input[77];
    input[79] += input[78];
    input[80] += input[79];
    input[81] += input[80];
    input[82] += input[81];
    input[83] += input[82];
    input[84] += input[83];
    input[85] += input[84];
    input[86] += input[85];
    input[87] += input[86];
    input[88] += input[87];
    input[89] += input[88];
    input[90] += input[89];
    input[91] += input[90];
    input[92] += input[91];
    input[93] += input[92];
    input[94] += input[93];
    input[95] += input[94];
    input[96] += input[95];
    input[97] += input[96];
    input[98] += input[97];
    input[99] += input[98];
    input[100] += input[99];
    input[101] += input[100];
    input[102] += input[101];
    input[103] += input[102];
    input[104] += input[103];
    input[105] += input[104];
    input[106] += input[105];
    input[107] += input[106];
    input[108] += input[107];
    input[109] += input[108];
    input[110] += input[109];
    input[111] += input[110];
    input[112] += input[111];
    input[113] += input[112];
    input[114] += input[113];
    input[115] += input[114];
    input[116] += input[115];
    input[117] += input[116];
    input[118] += input[117];
    input[119] += input[118];
    input[120] += input[119];
    input[121] += input[120];
    input[122] += input[121];
    input[123] += input[122];
    input[124] += input[123];
    input[125] += input[124];
    input[126] += input[125];
    input[127] += input[126];
  }

  private static void vectorPrefixSum(int[] input, int[] output) {
    IntVector vec0 = IntVector.fromArray(IntVector.SPECIES_128, input, 0);
    vec0 = vec0.add(vec0.unslice(1));
    vec0 = vec0.add(vec0.unslice(2));
    vec0.intoArray(output, 0);

    for (int off = 4; off < 128; off += IntVector.SPECIES_128.length()) {
      IntVector vec = IntVector.fromArray(IntVector.SPECIES_128, input, off);
      vec = vec.add(vec.unslice(1));
      vec = vec.add(vec.unslice(2));
      vec = vec.add(IntVector.broadcast(IntVector.SPECIES_128, output[off - 1]));
      vec.intoArray(output, off);
    }
  }

  private static void scalarDecodeTo32(long[] input, long[] output) {
    int shift = 0;
    int outBase = 0;
    for (int i = 0; i < 8; i++) {
      output[outBase] = (input[0] >>> shift) & LONG_MASK;
      output[outBase + 1] = (input[1] >>> shift) & LONG_MASK;
      output[outBase + 2] = (input[2] >>> shift) & LONG_MASK;
      output[outBase + 3] = (input[3] >>> shift) & LONG_MASK;
      output[outBase + 4] = (input[4] >>> shift) & LONG_MASK;
      output[outBase + 5] = (input[5] >>> shift) & LONG_MASK;
      output[outBase + 6] = (input[6] >>> shift) & LONG_MASK;
      output[outBase + 7] = (input[7] >>> shift) & LONG_MASK;
      shift += 4;
      outBase += 8;
    }
  }

  private static void vectorDecodeTo32_512(long[] input, long[] output) {
    LongVector inVec = LongVector.fromArray(L_SPECIES_512, input, 0);

    LongVector outVec = inVec.and(LONG_MASK);
    outVec.intoArray(output, 0);

    outVec = inVec.lanewise(VectorOperators.LSHR, 4).and(LONG_MASK);
    outVec.intoArray(output, 8);

    outVec = inVec.lanewise(VectorOperators.LSHR, 8).and(LONG_MASK);
    outVec.intoArray(output, 16);

    outVec = inVec.lanewise(VectorOperators.LSHR, 12).and(LONG_MASK);
    outVec.intoArray(output, 24);

    outVec = inVec.lanewise(VectorOperators.LSHR, 16).and(LONG_MASK);
    outVec.intoArray(output, 32);

    outVec = inVec.lanewise(VectorOperators.LSHR, 20).and(LONG_MASK);
    outVec.intoArray(output, 40);

    outVec = inVec.lanewise(VectorOperators.LSHR, 24).and(LONG_MASK);
    outVec.intoArray(output, 48);

    outVec = inVec.lanewise(VectorOperators.LSHR, 28).and(LONG_MASK);
    outVec.intoArray(output, 56);
  }

  private static void vectorDecodeTo32(long[] input, long[] output) {
    LongVector inVec;
    LongVector outVec;

    int inOff = 0;
    int outOff = 0;

    for (int i = 0; i < 4; i++) {
      inVec = LongVector.fromArray(L_SPECIES_128, input, inOff);

      outVec = inVec.and(LONG_MASK);
      outVec.intoArray(output, outOff);

      int outOffDelta = 8;
      int shift = 4;
      for (int j = 1; j < 8; j++) {
        outVec = inVec.lanewise(VectorOperators.LSHR, shift).and(LONG_MASK);
        outVec.intoArray(output, outOff + outOffDelta);

        outOffDelta += 8;
        shift += 4;
      }

      inOff += 2;
      outOff += 2;
    }
  }

  private static void scalarPrefixSum32(long[] input, int[] output) {
    input[1] += input[0];
    input[2] += input[1];
    input[3] += input[2];
    input[4] += input[3];
    input[5] += input[4];
    input[6] += input[5];
    input[7] += input[6];
    input[8] += input[7];
    input[9] += input[8];
    input[10] += input[9];
    input[11] += input[10];
    input[12] += input[11];
    input[13] += input[12];
    input[14] += input[13];
    input[15] += input[14];
    input[16] += input[15];
    input[17] += input[16];
    input[18] += input[17];
    input[19] += input[18];
    input[20] += input[19];
    input[21] += input[20];
    input[22] += input[21];
    input[23] += input[22];
    input[24] += input[23];
    input[25] += input[24];
    input[26] += input[25];
    input[27] += input[26];
    input[28] += input[27];
    input[29] += input[28];
    input[30] += input[29];
    input[31] += input[30];
    input[32] += input[31];
    input[33] += input[32];
    input[34] += input[33];
    input[35] += input[34];
    input[36] += input[35];
    input[37] += input[36];
    input[38] += input[37];
    input[39] += input[38];
    input[40] += input[39];
    input[41] += input[40];
    input[42] += input[41];
    input[43] += input[42];
    input[44] += input[43];
    input[45] += input[44];
    input[46] += input[45];
    input[47] += input[46];
    input[48] += input[47];
    input[49] += input[48];
    input[50] += input[49];
    input[51] += input[50];
    input[52] += input[51];
    input[53] += input[52];
    input[54] += input[53];
    input[55] += input[54];
    input[56] += input[55];
    input[57] += input[56];
    input[58] += input[57];
    input[59] += input[58];
    input[60] += input[59];
    input[61] += input[60];
    input[62] += input[61];
    input[63] += input[62];

    for (int i = 0; i < 64; ++i) {
      long l = input[i];
      output[i] = (int) (l >>> 32);
      output[64 + i] = (int) (l & 0xFFFFFFFFL);
    }

    int delta = output[63];
    for (int i = 64; i < 128; i++) {
      output[i] += delta;
    }
  }


//  @Benchmark
  public void prefixSumScalar(PrefixSumState state, Blackhole bh) {
    int[] input = state.input;
    int[] output = state.output;

    output[0] = input[0];
    for (int i = 1; i < input.length; ++i) {
      output[i] = output[i-1] + input[i];
    }

    bh.consume(output);
  }

//  @Benchmark
  public void prefixSumScalarInlined(PrefixSumState state, Blackhole bh) {
    int[] input = state.input;
    int[] output = state.output;

    output[0] = input[0];
    output[1] = output[0] + input[1];
    output[2] = output[1] + input[2];
    output[3] = output[2] + input[3];
    output[4] = output[3] + input[4];
    output[5] = output[4] + input[5];
    output[6] = output[5] + input[6];
    output[7] = output[6] + input[7];
    output[8] = output[7] + input[8];
    output[9] = output[8] + input[9];
    output[10] = output[9] + input[10];
    output[11] = output[10] + input[11];
    output[12] = output[11] + input[12];
    output[13] = output[12] + input[13];
    output[14] = output[13] + input[14];
    output[15] = output[14] + input[15];
    output[16] = output[15] + input[16];
    output[17] = output[16] + input[17];
    output[18] = output[17] + input[18];
    output[19] = output[18] + input[19];
    output[20] = output[19] + input[20];
    output[21] = output[20] + input[21];
    output[22] = output[21] + input[22];
    output[23] = output[22] + input[23];
    output[24] = output[23] + input[24];
    output[25] = output[24] + input[25];
    output[26] = output[25] + input[26];
    output[27] = output[26] + input[27];
    output[28] = output[27] + input[28];
    output[29] = output[28] + input[29];
    output[30] = output[29] + input[30];
    output[31] = output[30] + input[31];
    output[32] = output[31] + input[32];
    output[33] = output[32] + input[33];
    output[34] = output[33] + input[34];
    output[35] = output[34] + input[35];
    output[36] = output[35] + input[36];
    output[37] = output[36] + input[37];
    output[38] = output[37] + input[38];
    output[39] = output[38] + input[39];
    output[40] = output[39] + input[40];
    output[41] = output[40] + input[41];
    output[42] = output[41] + input[42];
    output[43] = output[42] + input[43];
    output[44] = output[43] + input[44];
    output[45] = output[44] + input[45];
    output[46] = output[45] + input[46];
    output[47] = output[46] + input[47];
    output[48] = output[47] + input[48];
    output[49] = output[48] + input[49];
    output[50] = output[49] + input[50];
    output[51] = output[50] + input[51];
    output[52] = output[51] + input[52];
    output[53] = output[52] + input[53];
    output[54] = output[53] + input[54];
    output[55] = output[54] + input[55];
    output[56] = output[55] + input[56];
    output[57] = output[56] + input[57];
    output[58] = output[57] + input[58];
    output[59] = output[58] + input[59];
    output[60] = output[59] + input[60];
    output[61] = output[60] + input[61];
    output[62] = output[61] + input[62];
    output[63] = output[62] + input[63];
    output[64] = output[63] + input[64];
    output[65] = output[64] + input[65];
    output[66] = output[65] + input[66];
    output[67] = output[66] + input[67];
    output[68] = output[67] + input[68];
    output[69] = output[68] + input[69];
    output[70] = output[69] + input[70];
    output[71] = output[70] + input[71];
    output[72] = output[71] + input[72];
    output[73] = output[72] + input[73];
    output[74] = output[73] + input[74];
    output[75] = output[74] + input[75];
    output[76] = output[75] + input[76];
    output[77] = output[76] + input[77];
    output[78] = output[77] + input[78];
    output[79] = output[78] + input[79];
    output[80] = output[79] + input[80];
    output[81] = output[80] + input[81];
    output[82] = output[81] + input[82];
    output[83] = output[82] + input[83];
    output[84] = output[83] + input[84];
    output[85] = output[84] + input[85];
    output[86] = output[85] + input[86];
    output[87] = output[86] + input[87];
    output[88] = output[87] + input[88];
    output[89] = output[88] + input[89];
    output[90] = output[89] + input[90];
    output[91] = output[90] + input[91];
    output[92] = output[91] + input[92];
    output[93] = output[92] + input[93];
    output[94] = output[93] + input[94];
    output[95] = output[94] + input[95];
    output[96] = output[95] + input[96];
    output[97] = output[96] + input[97];
    output[98] = output[97] + input[98];
    output[99] = output[98] + input[99];
    output[100] = output[99] + input[100];
    output[101] = output[100] + input[101];
    output[102] = output[101] + input[102];
    output[103] = output[102] + input[103];
    output[104] = output[103] + input[104];
    output[105] = output[104] + input[105];
    output[106] = output[105] + input[106];
    output[107] = output[106] + input[107];
    output[108] = output[107] + input[108];
    output[109] = output[108] + input[109];
    output[110] = output[109] + input[110];
    output[111] = output[110] + input[111];
    output[112] = output[111] + input[112];
    output[113] = output[112] + input[113];
    output[114] = output[113] + input[114];
    output[115] = output[114] + input[115];
    output[116] = output[115] + input[116];
    output[117] = output[116] + input[117];
    output[118] = output[117] + input[118];
    output[119] = output[118] + input[119];
    output[120] = output[119] + input[120];
    output[121] = output[120] + input[121];
    output[122] = output[121] + input[122];
    output[123] = output[122] + input[123];
    output[124] = output[123] + input[124];
    output[125] = output[124] + input[125];
    output[126] = output[125] + input[126];
    output[127] = output[126] + input[127];

    bh.consume(output);
  }

//  @Benchmark
  public void prefixSumVector128(PrefixSumState state, Blackhole bh) {
    int[] input = state.input;
    int[] output = state.output;

    IntVector vec0 = IntVector.fromArray(IntVector.SPECIES_128, input, 0);
    vec0 = vec0.add(vec0.unslice(1));
    vec0 = vec0.add(vec0.unslice(2));
    vec0.intoArray(output, 0);

    int upperBound = IntVector.SPECIES_128.loopBound(input.length);
    int i = IntVector.SPECIES_128.length();
    for (; i < upperBound; i += IntVector.SPECIES_128.length()) {
      IntVector vec = IntVector.fromArray(IntVector.SPECIES_128, input, i);
      vec = vec.add(vec.unslice(1));
      vec = vec.add(vec.unslice(2));
      vec = vec.add(IntVector.broadcast(IntVector.SPECIES_128, output[i-1]));
      vec.intoArray(output, i);
    }
    for (; i < input.length; ++i) {
      output[i] = output[i - 1] + input[i];
    }
    bh.consume(output);
  }

//  @Benchmark
  public void prefixSumVector256(PrefixSumState state, Blackhole bh) {
    int[] input = state.input;
    int[] output = state.output;

    IntVector vec0 = IntVector.fromArray(IntVector.SPECIES_256, input, 0);
    vec0 = vec0.add(vec0.unslice(1));
    vec0 = vec0.add(vec0.unslice(2));
    vec0 = vec0.add(vec0.unslice(4));
    vec0.intoArray(output, 0);

    int upperBound = IntVector.SPECIES_256.loopBound(input.length);
    int i = IntVector.SPECIES_256.length();
    for (; i < upperBound; i += IntVector.SPECIES_256.length()) {
      IntVector vec = IntVector.fromArray(IntVector.SPECIES_256, input, i);
      vec = vec.add(vec.unslice(1));
      vec = vec.add(vec.unslice(2));
      vec = vec.add(vec.unslice(4));
      vec = vec.add(IntVector.broadcast(IntVector.SPECIES_256, output[i-1]));
      vec.intoArray(output, i);
    }
    for (; i < input.length; ++i) {
      output[i] = output[i - 1] + input[i];
    }
    bh.consume(output);
  }

//  @Benchmark
  public void prefixSumVector512(PrefixSumState state, Blackhole bh) {
    int[] input = state.input;
    int[] output = state.output;

    IntVector vec0 = IntVector.fromArray(IntVector.SPECIES_512, input, 0);
    vec0 = vec0.add(vec0.unslice(1));
    vec0 = vec0.add(vec0.unslice(2));
    vec0 = vec0.add(vec0.unslice(4));
    vec0 = vec0.add(vec0.unslice(8));
    vec0.intoArray(output, 0);

    int upperBound = IntVector.SPECIES_512.loopBound(input.length);
    int i = IntVector.SPECIES_512.length();
    for (; i < upperBound; i += IntVector.SPECIES_512.length()) {
      IntVector vec = IntVector.fromArray(IntVector.SPECIES_512, input, i);
      vec = vec.add(vec.unslice(1));
      vec = vec.add(vec.unslice(2));
      vec = vec.add(vec.unslice(4));
      vec = vec.add(vec.unslice(8));
      vec = vec.add(IntVector.broadcast(IntVector.SPECIES_512, output[i-1]));
      vec.intoArray(output, i);
    }
    for (; i < input.length; ++i) {
      output[i] = output[i - 1] + input[i];
    }
    bh.consume(output);
  }

  private static final VectorShuffle<Integer> IOTA1_128 = VectorShuffle.iota(IntVector.SPECIES_128, -1, 1, true);
  private static final VectorShuffle<Integer> IOTA2_128 = VectorShuffle.iota(IntVector.SPECIES_128, -2, 1, true);
  private static final VectorMask<Integer> MASK1_128 = VectorMask.fromValues(IntVector.SPECIES_128, false, true, true, true);
  private static final VectorMask<Integer> MASK2_128 = VectorMask.fromValues(IntVector.SPECIES_128, false, false, true, true);

//  @Benchmark
  public void prefixSumVector128_v2(PrefixSumState state, Blackhole bh) {
    int[] input = state.input;
    int[] output = state.output;

    IntVector vec0 = IntVector.fromArray(IntVector.SPECIES_128, input, 0);
    vec0 = vec0.add(vec0.rearrange(IOTA1_128), MASK1_128);
    vec0 = vec0.add(vec0.rearrange(IOTA2_128), MASK2_128);
    vec0.intoArray(output, 0);

    int upperBound = IntVector.SPECIES_128.loopBound(input.length);
    int i = IntVector.SPECIES_128.length();
    for (; i < upperBound; i += IntVector.SPECIES_128.length()) {
      IntVector vec = IntVector.fromArray(IntVector.SPECIES_128, input, i);
      vec = vec.add(vec.rearrange(IOTA1_128), MASK1_128);
      vec = vec.add(vec.rearrange(IOTA2_128), MASK2_128);
      vec = vec.add(IntVector.broadcast(IntVector.SPECIES_128, output[i-1]));
      vec.intoArray(output, i);
    }
    for (; i < input.length; ++i) {
      output[i] = output[i - 1] + input[i];
    }
    bh.consume(output);
  }

  private static final VectorShuffle<Integer> IOTA1_256 = VectorShuffle.iota(IntVector.SPECIES_256, -1, 1, true);
  private static final VectorShuffle<Integer> IOTA2_256 = VectorShuffle.iota(IntVector.SPECIES_256, -2, 1, true);
  private static final VectorShuffle<Integer> IOTA4_256 = VectorShuffle.iota(IntVector.SPECIES_256, -4, 1, true);
  private static final VectorMask<Integer> MASK1_256 = VectorMask.fromValues(IntVector.SPECIES_256, false, true, true, true, true, true, true, true);
  private static final VectorMask<Integer> MASK2_256 = VectorMask.fromValues(IntVector.SPECIES_256, false, false, true, true, true, true, true, true);
  private static final VectorMask<Integer> MASK4_256 = VectorMask.fromValues(IntVector.SPECIES_256, false, false, false, false, true, true, true, true);


//  @Benchmark
  public void prefixSumVector256_v2(PrefixSumState state, Blackhole bh) {
    int[] input = state.input;
    int[] output = state.output;

    IntVector vec0 = IntVector.fromArray(IntVector.SPECIES_256, input, 0);
    vec0 = vec0.add(vec0.rearrange(IOTA1_256), MASK1_256);
    vec0 = vec0.add(vec0.rearrange(IOTA2_256), MASK2_256);
    vec0 = vec0.add(vec0.rearrange(IOTA4_256), MASK4_256);
    vec0.intoArray(output, 0);

    int upperBound = IntVector.SPECIES_256.loopBound(input.length);
    int i = IntVector.SPECIES_256.length();
    for (; i < upperBound; i += IntVector.SPECIES_256.length()) {
      IntVector vec = IntVector.fromArray(IntVector.SPECIES_256, input, i);
      vec = vec.add(vec.rearrange(IOTA1_256), MASK1_256);
      vec = vec.add(vec.rearrange(IOTA2_256), MASK2_256);
      vec = vec.add(vec.rearrange(IOTA4_256), MASK4_256);
      vec = vec.add(IntVector.broadcast(IntVector.SPECIES_256, output[i-1]));
      vec.intoArray(output, i);
    }
    for (; i < input.length; ++i) {
      output[i] = output[i - 1] + input[i];
    }
    bh.consume(output);
  }

  private static final VectorShuffle<Integer> IOTA1_256_EX = VectorShuffle.iota(IntVector.SPECIES_256, -1, 1, false);
  private static final VectorShuffle<Integer> IOTA2_256_EX = VectorShuffle.iota(IntVector.SPECIES_256, -2, 1, false);
  private static final VectorShuffle<Integer> IOTA4_256_EX = VectorShuffle.iota(IntVector.SPECIES_256, -4, 1, false);
  private static final IntVector ZERO_256 = IntVector.zero(IntVector.SPECIES_256);

//  @Benchmark
  public void prefixSumVector256_v3(PrefixSumState state, Blackhole bh) {
    int[] input = state.input;
    int[] output = state.output;

    IntVector vec0 = IntVector.fromArray(IntVector.SPECIES_256, input, 0);
    vec0 = vec0.add(vec0.rearrange(IOTA1_256_EX, ZERO_256));
    vec0 = vec0.add(vec0.rearrange(IOTA2_256_EX, ZERO_256));
    vec0 = vec0.add(vec0.rearrange(IOTA4_256_EX, ZERO_256));
    vec0.intoArray(output, 0);

    int upperBound = IntVector.SPECIES_256.loopBound(input.length);
    int i = IntVector.SPECIES_256.length();
    for (; i < upperBound; i += IntVector.SPECIES_256.length()) {
      IntVector vec = IntVector.fromArray(IntVector.SPECIES_256, input, i);
      vec = vec.add(vec.rearrange(IOTA1_256_EX, ZERO_256));
      vec = vec.add(vec.rearrange(IOTA2_256_EX, ZERO_256));
      vec = vec.add(vec.rearrange(IOTA4_256_EX, ZERO_256));
      vec = vec.add(IntVector.broadcast(IntVector.SPECIES_256, output[i-1]));
      vec.intoArray(output, i);
    }
    for (; i < input.length; ++i) {
      output[i] = output[i - 1] + input[i];
    }
    bh.consume(output);
  }

//  @Benchmark
  public void prefixSumVector256_v2_inline(PrefixSumState state, Blackhole bh) {
    int[] input = state.input;
    int[] output = state.output;

    IntVector vec = IntVector.fromArray(IntVector.SPECIES_256, input, 0);
    vec = vec.add(vec.rearrange(IOTA1_256), MASK1_256);
    vec = vec.add(vec.rearrange(IOTA2_256), MASK2_256);
    vec = vec.add(vec.rearrange(IOTA4_256), MASK4_256);
    vec.intoArray(output, 0);

    vec = IntVector.fromArray(IntVector.SPECIES_256, input, 8);
    vec = vec.add(vec.rearrange(IOTA1_256), MASK1_256);
    vec = vec.add(vec.rearrange(IOTA2_256), MASK2_256);
    vec = vec.add(vec.rearrange(IOTA4_256), MASK4_256);
    vec = vec.add(IntVector.broadcast(IntVector.SPECIES_256, output[7]));
    vec.intoArray(output, 8);

    vec = IntVector.fromArray(IntVector.SPECIES_256, input, 16);
    vec = vec.add(vec.rearrange(IOTA1_256), MASK1_256);
    vec = vec.add(vec.rearrange(IOTA2_256), MASK2_256);
    vec = vec.add(vec.rearrange(IOTA4_256), MASK4_256);
    vec = vec.add(IntVector.broadcast(IntVector.SPECIES_256, output[15]));
    vec.intoArray(output, 16);

    vec = IntVector.fromArray(IntVector.SPECIES_256, input, 24);
    vec = vec.add(vec.rearrange(IOTA1_256), MASK1_256);
    vec = vec.add(vec.rearrange(IOTA2_256), MASK2_256);
    vec = vec.add(vec.rearrange(IOTA4_256), MASK4_256);
    vec = vec.add(IntVector.broadcast(IntVector.SPECIES_256, output[23]));
    vec.intoArray(output, 24);

    vec = IntVector.fromArray(IntVector.SPECIES_256, input, 32);
    vec = vec.add(vec.rearrange(IOTA1_256), MASK1_256);
    vec = vec.add(vec.rearrange(IOTA2_256), MASK2_256);
    vec = vec.add(vec.rearrange(IOTA4_256), MASK4_256);
    vec = vec.add(IntVector.broadcast(IntVector.SPECIES_256, output[31]));
    vec.intoArray(output, 32);

    vec = IntVector.fromArray(IntVector.SPECIES_256, input, 40);
    vec = vec.add(vec.rearrange(IOTA1_256), MASK1_256);
    vec = vec.add(vec.rearrange(IOTA2_256), MASK2_256);
    vec = vec.add(vec.rearrange(IOTA4_256), MASK4_256);
    vec = vec.add(IntVector.broadcast(IntVector.SPECIES_256, output[39]));
    vec.intoArray(output, 40);

    vec = IntVector.fromArray(IntVector.SPECIES_256, input, 48);
    vec = vec.add(vec.rearrange(IOTA1_256), MASK1_256);
    vec = vec.add(vec.rearrange(IOTA2_256), MASK2_256);
    vec = vec.add(vec.rearrange(IOTA4_256), MASK4_256);
    vec = vec.add(IntVector.broadcast(IntVector.SPECIES_256, output[47]));
    vec.intoArray(output, 48);

    vec = IntVector.fromArray(IntVector.SPECIES_256, input, 56);
    vec = vec.add(vec.rearrange(IOTA1_256), MASK1_256);
    vec = vec.add(vec.rearrange(IOTA2_256), MASK2_256);
    vec = vec.add(vec.rearrange(IOTA4_256), MASK4_256);
    vec = vec.add(IntVector.broadcast(IntVector.SPECIES_256, output[55]));
    vec.intoArray(output, 56);

    vec = IntVector.fromArray(IntVector.SPECIES_256, input, 64);
    vec = vec.add(vec.rearrange(IOTA1_256), MASK1_256);
    vec = vec.add(vec.rearrange(IOTA2_256), MASK2_256);
    vec = vec.add(vec.rearrange(IOTA4_256), MASK4_256);
    vec = vec.add(IntVector.broadcast(IntVector.SPECIES_256, output[63]));
    vec.intoArray(output, 64);

    vec = IntVector.fromArray(IntVector.SPECIES_256, input, 72);
    vec = vec.add(vec.rearrange(IOTA1_256), MASK1_256);
    vec = vec.add(vec.rearrange(IOTA2_256), MASK2_256);
    vec = vec.add(vec.rearrange(IOTA4_256), MASK4_256);
    vec = vec.add(IntVector.broadcast(IntVector.SPECIES_256, output[71]));
    vec.intoArray(output, 72);

    vec = IntVector.fromArray(IntVector.SPECIES_256, input, 80);
    vec = vec.add(vec.rearrange(IOTA1_256), MASK1_256);
    vec = vec.add(vec.rearrange(IOTA2_256), MASK2_256);
    vec = vec.add(vec.rearrange(IOTA4_256), MASK4_256);
    vec = vec.add(IntVector.broadcast(IntVector.SPECIES_256, output[79]));
    vec.intoArray(output, 80);

    vec = IntVector.fromArray(IntVector.SPECIES_256, input, 88);
    vec = vec.add(vec.rearrange(IOTA1_256), MASK1_256);
    vec = vec.add(vec.rearrange(IOTA2_256), MASK2_256);
    vec = vec.add(vec.rearrange(IOTA4_256), MASK4_256);
    vec = vec.add(IntVector.broadcast(IntVector.SPECIES_256, output[87]));
    vec.intoArray(output, 88);

    vec = IntVector.fromArray(IntVector.SPECIES_256, input, 96);
    vec = vec.add(vec.rearrange(IOTA1_256), MASK1_256);
    vec = vec.add(vec.rearrange(IOTA2_256), MASK2_256);
    vec = vec.add(vec.rearrange(IOTA4_256), MASK4_256);
    vec = vec.add(IntVector.broadcast(IntVector.SPECIES_256, output[95]));
    vec.intoArray(output, 96);

    vec = IntVector.fromArray(IntVector.SPECIES_256, input, 104);
    vec = vec.add(vec.rearrange(IOTA1_256), MASK1_256);
    vec = vec.add(vec.rearrange(IOTA2_256), MASK2_256);
    vec = vec.add(vec.rearrange(IOTA4_256), MASK4_256);
    vec = vec.add(IntVector.broadcast(IntVector.SPECIES_256, output[103]));
    vec.intoArray(output, 104);

    vec = IntVector.fromArray(IntVector.SPECIES_256, input, 112);
    vec = vec.add(vec.rearrange(IOTA1_256), MASK1_256);
    vec = vec.add(vec.rearrange(IOTA2_256), MASK2_256);
    vec = vec.add(vec.rearrange(IOTA4_256), MASK4_256);
    vec = vec.add(IntVector.broadcast(IntVector.SPECIES_256, output[111]));
    vec.intoArray(output, 112);

    vec = IntVector.fromArray(IntVector.SPECIES_256, input, 120);
    vec = vec.add(vec.rearrange(IOTA1_256), MASK1_256);
    vec = vec.add(vec.rearrange(IOTA2_256), MASK2_256);
    vec = vec.add(vec.rearrange(IOTA4_256), MASK4_256);
    vec = vec.add(IntVector.broadcast(IntVector.SPECIES_256, output[119]));
    vec.intoArray(output, 120);

    bh.consume(output);
  }

  private static final VectorShuffle<Integer> IOTA1_512 = VectorShuffle.iota(IntVector.SPECIES_512, -1, 1, true);
  private static final VectorShuffle<Integer> IOTA2_512 = VectorShuffle.iota(IntVector.SPECIES_512, -2, 1, true);
  private static final VectorShuffle<Integer> IOTA4_512 = VectorShuffle.iota(IntVector.SPECIES_512, -4, 1, true);
  private static final VectorShuffle<Integer> IOTA8_512 = VectorShuffle.iota(IntVector.SPECIES_512, -8, 1, true);
  private static final VectorMask<Integer> MASK1_512 = VectorMask.fromValues(IntVector.SPECIES_512, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true);
  private static final VectorMask<Integer> MASK2_512 = VectorMask.fromValues(IntVector.SPECIES_512, false, false, true, true, true, true, true, true, true, true, true, true, true, true, true, true);
  private static final VectorMask<Integer> MASK4_512 = VectorMask.fromValues(IntVector.SPECIES_512, false, false, false, false, true, true, true, true, true, true, true, true, true, true, true, true);
  private static final VectorMask<Integer> MASK8_512 = VectorMask.fromValues(IntVector.SPECIES_512, false, false, false, false, false, false, false, false, true, true, true, true, true, true, true, true);


//  @Benchmark
  public void prefixSumVector512_v2(PrefixSumState state, Blackhole bh) {
    int[] input = state.input;
    int[] output = state.output;

    IntVector vec0 = IntVector.fromArray(IntVector.SPECIES_512, input, 0);
    vec0 = vec0.add(vec0.rearrange(IOTA1_512), MASK1_512);
    vec0 = vec0.add(vec0.rearrange(IOTA2_512), MASK2_512);
    vec0 = vec0.add(vec0.rearrange(IOTA4_512), MASK4_512);
    vec0 = vec0.add(vec0.rearrange(IOTA8_512), MASK8_512);
    vec0.intoArray(output, 0);

    int upperBound = IntVector.SPECIES_512.loopBound(input.length);
    int i = IntVector.SPECIES_512.length();
    for (; i < upperBound; i += IntVector.SPECIES_512.length()) {
      IntVector vec = IntVector.fromArray(IntVector.SPECIES_512, input, i);
      vec = vec.add(vec.rearrange(IOTA1_512), MASK1_512);
      vec = vec.add(vec.rearrange(IOTA2_512), MASK2_512);
      vec = vec.add(vec.rearrange(IOTA4_512), MASK4_512);
      vec = vec.add(vec.rearrange(IOTA8_512), MASK8_512);
      vec = vec.add(IntVector.broadcast(IntVector.SPECIES_512, output[i-1]));
      vec.intoArray(output, i);
    }
    for (; i < input.length; ++i) {
      output[i] = output[i - 1] + input[i];
    }
    bh.consume(output);
  }

  public void sanity() {
    var bh = new Blackhole("Today's password is swordfish. I understand instantiating Blackholes directly is dangerous.");

    var state = new PrefixSumState();
    state.setup();
    prefixSumScalar(state, bh);
    int[] expectedOutput = state.output;

    assertEqual(expectedOutput, this::scalarDecode_ScalarPrefixSum, bh);
    assertEqual(expectedOutput, this::vectorDecode_ScalarPrefixSum, bh);
    assertEqual(expectedOutput, this::vectorDecode512_ScalarPrefixSum, bh);
    assertEqual(expectedOutput, this::scalarDecodeFlex_ScalarPrefixSum, bh);
    assertEqual(expectedOutput, this::vectorDecodeFlex_ScalarPrefixSum, bh);
    assertEqual(expectedOutput, this::scalarDecode_VectorPrefixSum, bh);
    assertEqual(expectedOutput, this::vectorDecode_VectorPrefixSum_TwoPhase, bh);
    assertEqual(expectedOutput, this::scalarDecodeTo32_ScalarPrefixSum, bh);
    assertEqual(expectedOutput, this::vectorDecodeTo32_ScalarPrefixSum, bh);
    assertEqual(expectedOutput, this::vectorDecodeTo32_512_ScalarPrefixSum, bh);
//    assertEqual(expectedOutput, this::scalarDecodeTo32_VectorPrefixSum, bh);
//    assertEqual(expectedOutput, this::vectorDecodeTo32_VectorPrefixSum, bh);

    assertEqual(expectedOutput, this::prefixSumVector128, bh);
    assertEqual(expectedOutput, this::prefixSumVector128_v2, bh);
    assertEqual(expectedOutput, this::prefixSumVector256, bh);
    assertEqual(expectedOutput, this::prefixSumVector256_v2, bh);
    assertEqual(expectedOutput, this::prefixSumVector256_v3, bh);
    assertEqual(expectedOutput, this::prefixSumVector512, bh);
    assertEqual(expectedOutput, this::prefixSumVector512_v2, bh);
    assertEqual(expectedOutput, this::prefixSumScalarInlined, bh);
    assertEqual(expectedOutput, this::prefixSumVector256_v2_inline, bh);
  }

  static void assertEqual(int[] expectedOutput, BiConsumer<PrefixSumState, Blackhole> func, Blackhole bh) {
    var state = new PrefixSumState();
    state.setup();
    func.accept(state, bh);
    if (Arrays.equals(expectedOutput, state.output) == false) {
      throw new AssertionError("not equal: expected:\n" + Arrays.toString(expectedOutput) + ", got:\n" + Arrays.toString(state.output));
    }
  }
}
