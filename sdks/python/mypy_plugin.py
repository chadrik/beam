
from typing import Callable, Optional

from mypy.plugin import (
    Plugin, MethodContext
)
from mypy.types import (
    Type, Instance, TypeVarType, UninhabitedType
)
from mypy.expandtype import (
    expand_type
)


class BeamPlugin(Plugin):
    def get_method_hook(self, fullname: str
                        ) -> Optional[Callable[[MethodContext], Type]]:
        if fullname in {'apache_beam.pvalue.PValue.__or__',
                        'apache_beam.pvalue.PCollection.__or__'}:
            return pvalue_or_callback
        return None


def pvalue_or_callback(ctx: MethodContext) -> Type:
    """
    Callback to provide an accurate return type for
    apache_beam.pvalue.PValue.__or__.
    """
    if isinstance(ctx.type, Instance):
        assert len(ctx.arg_types) == 1, \
            'apache_beam.pvalue.PValue.__or__ should have exactly one parameter'
        assert len(ctx.arg_types[0]) == 1, \
            "apache_beam.pvalue.PValue.__or__'s parameter should not be variadic"
        xform_type = ctx.arg_types[0][0]

        xform_args = xform_type.args
        if len(xform_args) == 2:
            print("{}.__or__()".format(ctx.type))
            print("   default return: {}".format(ctx.default_return_type))
            print("   xform arg: {}".format(xform_type))
            print("   line: {}".format(ctx.context.line))

            pvalue = ctx.type
            # this is the PValue InT TypeVar T`1
            pvalue_typevar = pvalue.type.bases[0].args[0]

            # xform_typevar = xform_type.type
            # print("   {}".format(xform_typevar))

            # ths is the PTransform InT arg T`-1 (may be TypeVar or may be filled)
            xform_in_arg = xform_args[0]
            xform_out_arg = xform_args[1]

            # in_arg may already be filled, so we need the original TypeVarId
            meth = pvalue.type.get('__or__').type
            # print("   {}".format(meth))
            # print("   {}".format(meth.arg_types[0]))

            # print("   {}".format(ctx.arg_types))
            print("   pvalue typevar {}".format(pvalue_typevar))
            print("   xform typevar {} ({})".format(xform_in_arg, type(xform_in_arg)))

            if isinstance(xform_in_arg, TypeVarType):
                xform_expanded = expand_type(
                    xform_type, {xform_in_arg.id: pvalue.args[0]})
                out_arg = xform_expanded.args[1]
                print("   -> {}".format(xform_expanded))
                result = ctx.default_return_type.copy_modified(
                    args=[out_arg])
                print("   -> {}".format(result))
                return result
            elif isinstance(xform_in_arg, UninhabitedType) and isinstance(xform_out_arg, UninhabitedType):
                # print("   {}".format(xform_type.type))
                xform_type_inhabited = xform_type.copy_modified(
                    args=xform_type.type.bases[0].args)
                print("   {}".format(xform_type_inhabited))
            # else:
            #
            #     print("   {}".format(ctx.type.type.type_vars[0]))
            #     print("   {}".format(typevar))
            #     print("   {}".format(type(typevar.id)))
            #     print("   {}".format(in_arg))
            #     print("   {}".format(type(in_arg)))
            #     if isinstance(in_arg, Instance):
            #         print("   {}".format(in_arg.args))

            # for arg_type in ctx.arg_types:
            #     print("   {}".format(arg_type))
    return ctx.default_return_type


def plugin(version: str):
    # ignore version argument if the plugin works with all mypy versions.
    return BeamPlugin
