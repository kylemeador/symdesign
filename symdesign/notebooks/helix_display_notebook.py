import inspect
import os.path
import sys

import ipywidgets as widgets

from symdesign import flags
from symdesign.SymDesign import app

# All modules
# flags.available_modules
notebook_allowed_modules = [
    flags.align_helices,
    flags.analysis,
    flags.check_clashes,
    flags.cluster_poses,
    flags.design,
    flags.expand_asu,
    flags.generate_fragments,
    flags.helix_bending,
    flags.protocol,
    flags.rename_chains,
    flags.select_designs,
    flags.select_poses,
    flags.select_designs,
]

module_tags = widgets.TagsInput(
    value=[flags.align_helices],
    allowed_tags=notebook_allowed_modules,
    allow_duplicates=False
)
display(module_tags)


def get_class_that_defined_method(meth):
    """Where meth is a bound method"""
    if inspect.ismethod(meth):
        for cls in inspect.getmro(meth.__self__.__class__):
            if meth.__name__ in cls.__dict__:
                return cls
    return None


choices_widget = widgets.Dropdown
multiple_text = widgets.TagsInput
single_text = widgets.Text
# Add each requested as its own widget, formatting module arguments
all_module_widgets = []  # widgets.IntSlider(), widgets.Text()]
# for group in parser._action_groups:
for group in flags.argparsers[flags.parser_entire]._action_groups:
    for arg in group._group_actions:
        if isinstance(arg, argparse._SubParsersAction):
            for module, subparser in arg.choices.items():
                if module in module_tags.value:
                    module_description = widgets.Label(value=subparser.description)
                    module_widgets = [module_description]
                    for group in subparser._action_groups:
                        # These are the processed module arguments
                        for module_arg in group._group_actions:
                            # description = f'--{module_arg.dest}'
                            help = module_arg.help
                            choices = module_arg.choices
                            required = module_arg.required
                            type_ = module_arg.type
                            widget_kwargs = dict(value=module_arg.default,
                                                 description=module_arg.option_strings[-1],
                                                 tooltip=module_arg.help)
                            if isinstance(module_arg, (argparse._StoreTrueAction, argparse._StoreFalseAction,
                                                       argparse.BooleanOptionalAction)):  # .action in boolean_actions:
                                widget = widgets.ToggleButton(**widget_kwargs, tooltip=help)
                            elif isinstance(module_arg, argparse._StoreAction):
                                if choices:
                                    widget = choices_widget(**widget_kwargs, options=choices)
                                elif type_ is None or get_class_that_defined_method(type_) == str:
                                    # These are processed as strings
                                    if module_arg.nargs:
                                        widget = multiple_text(**widget_kwargs)
                                    else:
                                        widget = single_text(**widget_kwargs, placeholder=module_arg.metavar)
                                elif type_ == int:
                                    if module_arg.nargs:
                                        # There are currently none of these
                                        widget = widgets.IntsInput(**widget_kwargs)
                                    else:
                                        widget = widgets.IntText(**widget_kwargs)
                                elif type_ == float:
                                    if module_arg.nargs:
                                        # There are currently none of these
                                        widget = widgets.FloatsInput(**widget_kwargs)
                                    else:
                                        widget = widgets.FloatText(**widget_kwargs)
                                    widget = widgets.HBox([widget, widgets.Label(value=module_arg.metavar)])
                                elif type_ == os.path.abspath:
                                    if module_arg.nargs:
                                        # There are currently none of these
                                        widget = multiple_text(**widget_kwargs, placeholder=module_arg.metavar)
                                    else:
                                        widget = single_text(**widget_kwargs, placeholder=module_arg.metavar)
                                # Custom types
                                elif type_ == flags.temp_gt0:
                                    widget = widgets.FloatsInput(**widget_kwargs)
                            else:
                                continue
                            # widgets.HBox([widget, widgets.Label(value=module_arg.help)])
                            # Add the module widgets to all widgets
                            module_widgets.append(widget)
                    # Add the module widgets to all widgets
                    all_module_widgets.append(widgets.VBox(module_widgets))
# For formatting the help, use
# widgets.HBox([WIDGET, widgets.Label(value="The $m$ in $E=mc^2$:")])
# for boolean flags, use
# widgets.ToggleButton(
#     value=False,
#     description='Click me',
#     disabled=False,
#     button_style='', # 'success', 'info', 'warning', 'danger' or ''
#     tooltip='Description',
#     icon='check' # (FontAwesome names without the `fa-` prefix)
# )
# For choices flags, use
# widgets.Dropdown(
#     options=[('One', 1), ('Two', 2), ('Three', 3)],
#     # options=['1', '2', '3'],
#     value=2,
#     description='Number:',
# )
# For int flags, use
# widgets.IntText(
#     value=7,
#     description='Any:',
#     disabled=False
# )
# widgets.IntsInput(
#     value=[1, 4, 3243],
#     min=0,
#     max=1000000,
#     format='$,d'
# )
# For float flags, use
# widgets.FloatText(
#     value=7,
#     description='Any:',
#     disabled=False
# )
# For string flags, use
# widgets.Text(
#     value='Hello World',
#     placeholder='Type something',
#     description='String:',
#     disabled=False
# )

# ArgumentParser(prog='\n      python SymDesign.py generate-fragments [input arguments][output arguments][design selector arguments][optional arguments]', usage=None, description='Generate fragment overlap for poses of interest and write fragments', formatter_class=<class 'symdesign.flags.Formatter'>, conflict_handler='error', add_help=True)
# Set up the display to show all options for each module
# # Accordian display
# options_accordion = widgets.Accordion(children=all_module_widgets, titles=module_tags.value)
# display(options_accordion)
# Tab display
tab = widgets.Tab()
tab.children = all_module_widgets
tab.titles = module_tags.value  # ['align', 'design']
display(options_tab)


# Format the arguments input from user into parsable command-line like arguments
if len(module_tags.value) > 1:
    sys.argv = ['protocol']
else:
    sys.argv = ['protocol', '--module'] + module_tags.value


run_protocol_button = widgets.Button(
    description='Run protocol',
    disabled=False,
    button_style='success',  # 'success', 'info', 'warning', 'danger' or ''
    tooltip='To perform the protocols specified, click this button',
    icon='turn-down-left'
)
run_protocol_button.on_click(app)
