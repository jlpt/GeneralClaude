OUTPUT_FORMAT("elf32-tradbigmips")
ENTRY(_start)
SECTIONS
{
    . = 0x80000400;
    .text : { *(.text.start) *(.text*) }
    .rodata : { *(.rodata*) }
    .data : { *(.data*) *(.sdata*) }
    . = ALIGN(8);
    __bss_start = .;
    .bss : { *(.sbss*) *(.scommon) *(.bss*) *(COMMON) }
    . = ALIGN(8);
    __bss_end = .;
    /DISCARD/ : { *(.MIPS.abiflags) *(.reginfo) *(.comment) *(.pdr) *(.gnu.attributes) *(.note*) }
}
