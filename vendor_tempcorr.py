#!/usr/bin/env python3
"""
Reference temperature correction: runs the vendor's enhance_distance_temp_correct
from libadvirtempac020.so (arm64, from the Vantrue Thermal APK) under Unicorn.

VendorTempCorrection.correct() takes and returns degrees C, with the same
arguments the app passes to LibIRTempAC020.temperatureCorrection for one gain.
app_display_c() applies the app's rules around it for one displayed value.

The vendor library and tau tables are read from vendor/ next to this file;
README.md describes extracting them from your own copy of the app.
"""
import math
import pathlib
import struct
from decimal import ROUND_FLOOR, Decimal

import unicorn as uc
from elftools.elf.elffile import ELFFile
from elftools.elf.relocation import RelocationSection
from unicorn import arm64_const as a64

VENDOR_DIR = pathlib.Path(__file__).parent / "vendor"
SO_PATH = VENDOR_DIR / "libadvirtempac020.so"
TABLE_DIR = VENDOR_DIR

BASE = 0x1000_0000
STUBS = 0x2000_0000
HEAP = 0x3000_0000
STACK = 0x4000_0000
TLS = 0x5000_0000
RET_MAGIC = 0x6000_0000
REGION = 0x10_0000
ARM64_RET = struct.pack("<I", 0xD65F03C0)

GAIN_LOW = 0
GAIN_HIGH = 1
TABLES = {
    GAIN_LOW: "V303_P3_4.3mm_L.bin",
    GAIN_HIGH: "V303_P3_4.3mm_H.bin",
}
GAIN_RANGE_C = {
    GAIN_LOW: (0.0, 550.0),
    GAIN_HIGH: (-20.0, 150.0),
}
LOW_GAIN_FLOOR_C = 150.0
APP_DEFAULTS = dict(ems=1.0, ta=25.0, tu=25.0, dist=0.25, hum=0.8)

R_AARCH64_ABS64 = 257
R_AARCH64_GLOB_DAT = 1025
R_AARCH64_JUMP_SLOT = 1026
R_AARCH64_RELATIVE = 1027


class VendorTempCorrection:
    """
    One loaded instance of the vendor library. Not thread-safe.
    """

    def __init__(self, so_path=SO_PATH):
        self.mu = uc.Uc(uc.UC_ARCH_ARM64, uc.UC_MODE_ARM)
        self.stub_names = []
        self.heap_top = HEAP

        with open(so_path, "rb") as f:
            elf = ELFFile(f)
            self.symbols = self._load(elf)

        for addr in (HEAP, STACK, TLS, RET_MAGIC):
            self.mu.mem_map(addr, REGION)
        self.mu.mem_map(STUBS, REGION)
        self.mu.mem_write(STUBS, ARM64_RET * len(self.stub_names))
        self.mu.hook_add(uc.UC_HOOK_CODE, self._stub_hook, begin=STUBS, end=STUBS + 4 * len(self.stub_names))

        self.mu.reg_write(a64.UC_ARM64_REG_CPACR_EL1, 3 << 20)
        self.mu.reg_write(a64.UC_ARM64_REG_TPIDR_EL0, TLS)

    # ---- loader ----------------------------------------------------------------

    def _load(self, elf):
        loads = [s for s in elf.iter_segments() if s["p_type"] == "PT_LOAD"]
        span = max(s["p_vaddr"] + s["p_memsz"] for s in loads)
        self.mu.mem_map(BASE, (span + REGION - 1) // REGION * REGION)
        for s in loads:
            self.mu.mem_write(BASE + s["p_vaddr"], s.data())

        dynsym = elf.get_section_by_name(".dynsym")
        symbols = {s.name: BASE + s["st_value"] for s in dynsym.iter_symbols() if s["st_shndx"] != "SHN_UNDEF"}

        for sec in elf.iter_sections():
            if not isinstance(sec, RelocationSection):
                continue

            symtab = elf.get_section(sec["sh_link"])
            for r in sec.iter_relocations():
                kind = r["r_info_type"]
                if kind == R_AARCH64_RELATIVE:
                    value = BASE + r["r_addend"]
                elif kind in (R_AARCH64_ABS64, R_AARCH64_GLOB_DAT, R_AARCH64_JUMP_SLOT):
                    sym = symtab.get_symbol(r["r_info_sym"])
                    target = symbols.get(sym.name) or self._stub(sym.name)
                    value = target + r["r_addend"]
                else:
                    raise NotImplementedError(f"relocation type {kind} at {r['r_offset']:#x}")
                self.mu.mem_write(BASE + r["r_offset"], struct.pack("<Q", value))

        return symbols

    def _stub(self, name):
        if name not in self.stub_names:
            self.stub_names.append(name)
        return STUBS + 4 * self.stub_names.index(name)

    # ---- libc stand-ins --------------------------------------------------------

    def _x(self, n):
        return self.mu.reg_read(a64.UC_ARM64_REG_X0 + n)

    def _d(self, n):
        return struct.unpack("<d", struct.pack("<Q", self.mu.reg_read(a64.UC_ARM64_REG_D0 + n)))[0]

    def _ret_d(self, v):
        self.mu.reg_write(a64.UC_ARM64_REG_D0, struct.unpack("<Q", struct.pack("<d", v))[0])

    def _alloc(self, size):
        addr = self.heap_top
        self.heap_top += (size + 15) & ~15
        if self.heap_top > HEAP + REGION:
            raise MemoryError("emulated heap exhausted")
        self.mu.mem_write(addr, bytes(size))
        return addr

    def _stub_hook(self, mu, address, size, user_data):
        name = self.stub_names[(address - STUBS) // 4]
        x0 = 0

        if name == "exp":
            return self._ret_d(math.exp(self._d(0)))
        if name == "log":
            return self._ret_d(math.log(self._d(0)))
        if name == "pow":
            return self._ret_d(math.pow(self._d(0), self._d(1)))
        if name == "memcpy":
            mu.mem_write(self._x(0), bytes(mu.mem_read(self._x(1), self._x(2))))
            x0 = self._x(0)
        elif name == "memset":
            mu.mem_write(self._x(0), bytes([self._x(1) & 0xFF]) * self._x(2))
            x0 = self._x(0)
        elif name == "malloc":
            x0 = self._alloc(self._x(0))
        elif name == "calloc":
            x0 = self._alloc(self._x(0) * self._x(1))
        elif name == "basename":
            x0 = self._x(0)
        elif name == "gettid":
            x0 = 1
        elif name == "__stack_chk_fail":
            raise RuntimeError("vendor code hit __stack_chk_fail")
        mu.reg_write(a64.UC_ARM64_REG_X0, x0)

    # ---- entry point -----------------------------------------------------------

    def correct(self, temp_c, table, ems, ta, tu, dist, hum):
        """
        Corrected object temperature in C, or raises ValueError with the vendor
        return code when the vendor code rejects the inputs.
        """
        self.heap_top = HEAP
        env = self._alloc(20)
        tab = self._alloc(len(table))
        out = self._alloc(4)
        self.mu.mem_write(env, struct.pack("<5f", dist, ems, hum, ta, tu))
        self.mu.mem_write(tab, bytes(table))

        self.mu.reg_write(a64.UC_ARM64_REG_S0, struct.unpack("<I", struct.pack("<f", temp_c))[0])
        self.mu.reg_write(a64.UC_ARM64_REG_X0, env)
        self.mu.reg_write(a64.UC_ARM64_REG_X1, tab)
        self.mu.reg_write(a64.UC_ARM64_REG_X2, 8)
        self.mu.reg_write(a64.UC_ARM64_REG_X3, out)
        self.mu.reg_write(a64.UC_ARM64_REG_SP, STACK + REGION - 0x100)
        self.mu.reg_write(a64.UC_ARM64_REG_X30, RET_MAGIC)
        self.mu.emu_start(self.symbols["enhance_distance_temp_correct"], RET_MAGIC)

        rc = struct.unpack("<i", struct.pack("<I", self.mu.reg_read(a64.UC_ARM64_REG_X0) & 0xFFFFFFFF))[0]
        if rc != 0:
            raise ValueError(f"vendor rc {rc}")
        return struct.unpack("<f", self.mu.mem_read(out, 4))[0]


def load_tables(table_dir=TABLE_DIR):
    return {gain: (table_dir / name).read_bytes() for gain, name in TABLES.items()}


def app_display_c(vc, tables, camera_c, gain, ems, ta, tu, dist, hum):
    """
    The value the Vantrue app displays for one measurement, in C, or None where
    it displays "<150 C". camera_c is the camera's raw / 64 - 273.15.

    Mirrors TempSetHelper.getCompensateTempById for a P2L-class device and the
    low-gain rule of TempSetHelper.handleTempForCurrentUnit.
    """
    lo, hi = GAIN_RANGE_C[gain]
    t = camera_c
    if lo <= camera_c <= hi:
        try:
            t = vc.correct(camera_c, tables[gain], ems, ta, tu, dist, hum)
        except ValueError:
            t = camera_c

    if math.isnan(t) or abs(t) > 10000.0:
        t = camera_c
    else:
        t = float(Decimal(t).quantize(Decimal("0.01"), rounding=ROUND_FLOOR))

    if gain == GAIN_LOW and t < LOW_GAIN_FLOOR_C:
        return None
    return t
