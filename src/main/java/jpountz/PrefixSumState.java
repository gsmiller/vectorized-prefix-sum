package jpountz;

import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;

@State(Scope.Benchmark)
public class PrefixSumState {

//  @Param({"128", "1024"})
  final int size = 128;

  int[] input ;
  int[] output;

  int[] inputPackedInts;
  long[] inputPackedLongs;

  int[] inputPackedInts512;

  int[] inputPackedIntsFlex;

  @Setup(Level.Trial)
  public void setup() {
    final int max = (1 << 4) - 1; // 0x0F (0 - 15)
    final int bpv = 4;

    input = new int[size];
    output = new int[size];
    for (int i = 0; i < input.length; i++) {
      input[i] = (i + 1) & max;
    }

    inputPackedInts = new int[4 * bpv];
    pack(input, inputPackedInts);

    inputPackedLongs = new long[2 * bpv];
    pack(input, inputPackedLongs);

    inputPackedInts512 = new int[4 * bpv];
    pack512(input, inputPackedInts512);

    inputPackedIntsFlex = new int[4 * bpv];
    packFlex(input, inputPackedIntsFlex);
  }

  private static final VectorSpecies<Integer> SPECIES_128 = IntVector.SPECIES_128;
  private static final VectorSpecies<Long> L_SPECIES_128 = LongVector.SPECIES_128;

  private static final VectorSpecies<Integer> SPECIES_512 = IntVector.SPECIES_512;

  private static final VectorSpecies<Integer> SPECIES_FLEX = IntVector.SPECIES_PREFERRED;

  private static void pack(int[] input, int[] output) {
    assert input.length == 128;
    assert output.length == 16;

    int inOff = 0;
    int outOff = 0;
    IntVector outVec;
    IntVector inVec;

    for (int i = 0; i < 4; i++) {
      inVec = IntVector.fromArray(SPECIES_128, input, inOff);
      outVec = inVec;

      inVec = IntVector.fromArray(SPECIES_128, input, inOff + 4);
      outVec = inVec.lanewise(VectorOperators.LSHL, 4).or(outVec);

      inVec = IntVector.fromArray(SPECIES_128, input, inOff + 8);
      outVec = inVec.lanewise(VectorOperators.LSHL, 8).or(outVec);

      inVec = IntVector.fromArray(SPECIES_128, input, inOff + 12);
      outVec = inVec.lanewise(VectorOperators.LSHL, 12).or(outVec);

      inVec = IntVector.fromArray(SPECIES_128, input, inOff + 16);
      outVec = inVec.lanewise(VectorOperators.LSHL, 16).or(outVec);

      inVec = IntVector.fromArray(SPECIES_128, input, inOff + 20);
      outVec = inVec.lanewise(VectorOperators.LSHL, 20).or(outVec);

      inVec = IntVector.fromArray(SPECIES_128, input, inOff + 24);
      outVec = inVec.lanewise(VectorOperators.LSHL, 24).or(outVec);

      inVec = IntVector.fromArray(SPECIES_128, input, inOff + 28);
      outVec = inVec.lanewise(VectorOperators.LSHL, 28).or(outVec);

      outVec.intoArray(output, outOff);

      inOff += 32;
      outOff += 4;
    }
  }

  private static void pack512(int[] input, int[] output) {
    assert input.length == 128;
    assert output.length == 16;

    IntVector inVec = IntVector.fromArray(SPECIES_512, input, 0);
    IntVector outVec = inVec;

    inVec = IntVector.fromArray(SPECIES_512, input, 16);
    outVec = inVec.lanewise(VectorOperators.LSHL, 4).or(outVec);

    inVec = IntVector.fromArray(SPECIES_512, input, 32);
    outVec = inVec.lanewise(VectorOperators.LSHL, 8).or(outVec);

    inVec = IntVector.fromArray(SPECIES_512, input, 48);
    outVec = inVec.lanewise(VectorOperators.LSHL, 12).or(outVec);

    inVec = IntVector.fromArray(SPECIES_512, input, 64);
    outVec = inVec.lanewise(VectorOperators.LSHL, 16).or(outVec);

    inVec = IntVector.fromArray(SPECIES_512, input, 80);
    outVec = inVec.lanewise(VectorOperators.LSHL, 20).or(outVec);

    inVec = IntVector.fromArray(SPECIES_512, input, 96);
    outVec = inVec.lanewise(VectorOperators.LSHL, 24).or(outVec);

    inVec = IntVector.fromArray(SPECIES_512, input, 112);
    outVec = inVec.lanewise(VectorOperators.LSHL, 28).or(outVec);

    outVec.intoArray(output, 0);
  }

  private static void packFlex(int[] input, int[] output) {
    assert input.length == 128;
    assert output.length == 16;

    IntVector inVec;
    IntVector outVec;

    int off = 0;

    int lanes = SPECIES_FLEX.length();
    assert 16 % lanes == 0;
    int bound = 16 / lanes;
    for (int i = 0; i < bound; i++) {
      inVec = IntVector.fromArray(SPECIES_FLEX, input, off);
      outVec = inVec;

      inVec = IntVector.fromArray(SPECIES_FLEX, input, off + 16);
      outVec = inVec.lanewise(VectorOperators.LSHL, 4).or(outVec);

      inVec = IntVector.fromArray(SPECIES_FLEX, input, off + 32);
      outVec = inVec.lanewise(VectorOperators.LSHL, 8).or(outVec);

      inVec = IntVector.fromArray(SPECIES_FLEX, input, off + 48);
      outVec = inVec.lanewise(VectorOperators.LSHL, 12).or(outVec);

      inVec = IntVector.fromArray(SPECIES_FLEX, input, off + 64);
      outVec = inVec.lanewise(VectorOperators.LSHL, 16).or(outVec);

      inVec = IntVector.fromArray(SPECIES_FLEX, input, off + 80);
      outVec = inVec.lanewise(VectorOperators.LSHL, 20).or(outVec);

      inVec = IntVector.fromArray(SPECIES_FLEX, input, off + 96);
      outVec = inVec.lanewise(VectorOperators.LSHL, 24).or(outVec);

      inVec = IntVector.fromArray(SPECIES_FLEX, input, off + 112);
      outVec = inVec.lanewise(VectorOperators.LSHL, 28).or(outVec);

      outVec.intoArray(output, off);

      off += lanes;
    }
  }

  private static void pack(int[] input, long[] output) {
    assert input.length == 128;
    assert output.length == 8;

    long[] longs = new long[128];
    for (int i = 0; i < 128; i++) {
      longs[i] = input[i];
    }

    int inOff = 0;
    int outOff = 0;
    LongVector outVec;
    LongVector inVec;

    for (int i = 0; i < 4; i++) {
      inVec = LongVector.fromArray(L_SPECIES_128, longs, inOff + 64);
      outVec = inVec;

      inVec = LongVector.fromArray(L_SPECIES_128, longs, inOff);
      outVec = inVec.lanewise(VectorOperators.LSHL, 32).or(outVec);

      int inOffDelta = 8;
      int shift = 4;
      for (int j = 1; j < 8; j++) {
        inVec = LongVector.fromArray(L_SPECIES_128, longs, inOff + 64 + inOffDelta);
        outVec = inVec.lanewise(VectorOperators.LSHL, shift).or(outVec);

        inVec = LongVector.fromArray(L_SPECIES_128, longs, inOff + inOffDelta);
        outVec = inVec.lanewise(VectorOperators.LSHL, 32 + shift).or(outVec);

        inOffDelta += 8;
        shift += 4;
      }

      outVec.intoArray(output, outOff);
      inOff += 2;
      outOff += 2;
    }
  }
}
