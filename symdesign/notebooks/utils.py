from __future__ import annotations

import inspect
import argparse
import os
import subprocess
import sys
from collections.abc import Callable, Iterable
from typing import Any

import ipywidgets as widgets
from IPython.core.display_functions import display

from .. import flags
from ..run import app
from ..structure.utils import StructureException
from ..utils import SymDesignException

# Load and set up job distribution widgets
choices_widget = widgets.Dropdown
# This only works on ipywidgets > 8, which colab currently doesn't support...
# multiple_text = widgets.TagsInput
# START Hack all TagsInput
additional_value = '+ additional'
multiple_text_kwargs = dict(
    description=additional_value,
    disabled=False,
    button_style='',  # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Click to add another value',
    icon='turn-down-left',
    layout=widgets.Layout(width='auto')
)
additional_value_button_widget = widgets.Button(**multiple_text_kwargs)
multiple_text_row_layout = widgets.Layout(display='flex',
                                          flex_flow='row wrap',
                                          justify_content='flex-start',
                                          align_content='stretch',
                                          align_items='stretch'
                                          # border='solid',
                                          # width=f'100%'
                                          )


class MultipleTextBase(widgets.Widget):  # ABC
    base_widget = None
    button = None
    button_kwargs: dict = None
    subsequent_widget = None

    def __init__(self, subsequent_kwargs: dict[str, Any] = None, **kwargs):
        super().__init__()
        self._kwargs = kwargs
        self.button = widgets.Button(**self.__class__.button_kwargs)
        self.button.on_click(self.add_input)
        # Todo
        #  make the base_widget layout= kwarg dependent on a size, minimize self.button layout= also
        self.box = widgets.Box(
            [self.__class__.base_widget(**self._kwargs), self.button],
            layout=row_layout)
        if self.__class__.subsequent_widget is None:
            self.__class__.subsequent_widget = self.__class__.base_widget
            if subsequent_kwargs:
                self._kwargs = subsequent_kwargs

    # property
    # def values(self):
    #     print('API WANING: .values was used...')
    #     return [child.value for child in self.children]

    @property
    def value(self):
        return [child.value for child in self.children]

    @property
    def children(self):
        return list(self.box.children[:-1])

    def add_input(self, button):
        self.box.children = [*self.children,
                             self.__class__.subsequent_widget(**self._kwargs),
                             self.button]

    def _ipython_display_(self):
        display(self.box)


# Doesn't work in ipywidgets < 8
# module_tags = widgets.TagsInput(
#     value=[flags.align_helices],
#     allowed_tags=notebook_allowed_modules,
#     allow_duplicates=False
# )
# START 7.7.1 TagsInput hack
# module_selection_widget = widgets.Dropdown(value=flags.align_helices, options=notebook_allowed_modules)
additional_module = '+ module'
additional_module_tooltip = 'Click to add another module'
additional_module_kwargs = dict(
    description=additional_module,
    disabled=False,
    button_style='',  # 'success', 'info', 'warning', 'danger' or ''
    tooltip=additional_module_tooltip,
    icon='turn-down-left',
    layout=widgets.Layout(width='auto')
)
additional_module_button_widget = widgets.Button(**additional_module_kwargs)
# module_tags = widgets.Box(
#     [copy.copy(module_selection_widget), additional_module_button_widget],
#     layout=row_layout)
# def request_user_modules(*args):
#     module_tags.children = module_tags.children.insert(0, copy.copy(module_selection_widget))

# additional_module_button_widget.on_click(request_user_modules)


class MultipleModule(MultipleTextBase):
    base_widget = widgets.Dropdown
    button = additional_module_button_widget
    button_kwargs = additional_module_kwargs


class MultipleText(MultipleTextBase):
    base_widget = widgets.Text
    button = additional_value_button_widget
    button_kwargs = multiple_text_kwargs


# END hack
multiple_text = MultipleText
single_text = widgets.Text
boolean_widget = widgets.Checkbox  # widgets.ToggleButton
required_widget = widgets.Valid  # (value=False, description=None)
input_field_size = 80
input_field_layout = widgets.Layout(display='flex',
                                    flex_flow='row',
                                    width=f'{input_field_size / 4}%')
multitext_individual_layout = widgets.Layout(display='flex', width='auto')

# Use with [options-inputs row, ...] formating
row_layout = widgets.Layout(display='flex',
                            flex_flow='row',
                            # justify_content='space-between',
                            align_content='stretch',
                            align_items='stretch'
                            # border='solid',
                            # width='100%'
                            )
box_layout = widgets.Layout(display='flex',
                            flex_flow='column',
                            align_items='stretch',
                            border='solid',
                            width='100%')
# # Use with options column / inputs column formating
# row_layout = widgets.Layout(display='flex',
#                             flex_flow='column',
#                             justify_content='center',
#                             align_content='stretch',
#                             align_items='stretch'
#                             # border='solid',
#                             # width='100%'
#                            )
# description_row_layout = widgets.Layout(display='flex',
#                             flex_flow='column',
#                             justify_content='center',
#                             align_content='flex-end',
#                             align_items='flex-end',
#                             # border='solid',
#                             width='100%'
#                            )
# box_layout = widgets.Layout(display='flex',
#                             flex_flow='row',
#                             align_items='stretch',
#                             border='solid',
#                             width='100%')

description_layout = widgets.Layout(display='flex',
                                    justify_content='flex-end',
                                    width=f'{100 - input_field_size}%')
module_options_description = widgets.HTML(value='<b>Module description:</b>')


def process_module_arg_to_widget(arg, widget_kwargs):
    if arg.nargs:
        # START Hack all TagsInput
        if isinstance(widget_kwargs['value'], tuple):
            widget_kwargs['value'] = ''
        # END
        widget = multiple_text(**widget_kwargs,
                               layout=multitext_individual_layout)
        # START Hack all TagsInput
        widget = widget.box
        # END
    else:
        if arg.metavar:
            placeholder = arg.metavar
        else:
            placeholder = ''  # arg.default
        widget = single_text(**widget_kwargs, placeholder=placeholder,
                             layout=input_field_layout)

    return widget


def get_class_that_defined_method(meth: Callable) -> Any | None:
    """Where meth is a bound method of a class

    Args:
        meth: A callable that is potentially a method of another class
    Returns:
        The class of the method if it is a method or method descriptor
    """
    if inspect.ismethod(meth):
        # print('Found method', meth)
        for cls in inspect.getmro(meth.__self__.__class__):
            if meth.__name__ in cls.__dict__:
                return cls
    elif inspect.ismethoddescriptor(meth):
        # print('Found methoddescriptor', meth)
        try:
            return getattr(meth, '__objclass__')
        except AttributeError:
            print(f"Couldn't find the class (using '__objclass__') of the method descriptor {meth}\n")
            print(dir(meth))
            raise
    elif inspect.isfunction(meth):
        # print('Found function', meth)
        # Ex: os.path.abspath
        pass
    elif inspect.isclass(meth):
        # print('Found class', meth)
        # Ex: int, float, str
        pass
    else:
        print(f"Couldn't recognize the method {meth}")
        print(inspect.ismemberdescriptor(meth))
        # This doesn't exist for some reason
        #  inspect.ismethodwrapper(meth)
    return None


# def argparse_modules_to_ipywidgets() -> dict[str, widgets.Widget]:
#     """Format each known module to a widget view, formatting module arguments appropriately
#
#     Returns:
#         The mapping of each module name to the collection of widget instances which define its options
#     """

# Create ipywidgets
boolean_optional_flags = {}
all_module_widgets = {}
for group in flags.entire_parser._action_groups:
    for arg in group._group_actions:
        if isinstance(arg, argparse._SubParsersAction):
            for module, subparser in arg.choices.items():
                # try:
                #     module_index = module_tags.value.index(module)
                #     options_module = False
                # except ValueError:
                #     if module in flags.options_modules:
                #         # Parse these flags as they are options that the user can specify
                #         print(module)
                #         options_module = True
                #         extra_modules.append(module)
                #         module_index = number_of_modules + next(extra_module_count)
                #     else:
                #         # print('NOT module', module)
                #         continue

                # module_description = widgets.Label(value=subparser.description)
                module_description = widgets.HTML(value=subparser.description)
                module_widgets = [module_options_description, module_description]
                # # Use with options column / inputs column formatting
                # module_widgets = [module_description]
                # description_widgets = [module_options_description]
                for group in subparser._action_groups:
                    # These are the processed module arguments
                    for module_arg in group._group_actions:
                        # print(module_arg)
                        # help = module_arg.help
                        choices = module_arg.choices
                        # required = module_arg.required
                        # if required:
                        #     print('Required', module_arg.option_strings[-1])
                        type_ = module_arg.type

                        if module_arg.default is None:
                            if module_arg.nargs:
                                # print(module_arg.option_strings[-1], 'is None, but has nargs')
                                # print(module_arg)
                                module_value = None  # [] # None]
                                # choices = none_tuple + copy(choices)
                            else:
                                module_value = module_arg.default
                        elif isinstance(module_arg.default, bool):
                            module_value = module_arg.default
                        elif isinstance(module_arg.default, (int, float)):
                            module_value = str(module_arg.default)
                        else:
                            module_value = module_arg.default

                        # long_argument is format --ignore-clashes
                        long_argument = module_arg.option_strings[-1]
                        tooltip = module_arg.help % vars(module_arg)
                        widget_kwargs = dict(value=module_value,
                                             # description=module_arg.option_strings[-1],
                                             description_tooltip=tooltip,
                                             tooltip=tooltip
                                             )
                        if isinstance(module_arg,
                                      (argparse._StoreTrueAction,
                                       argparse._StoreFalseAction,
                                       argparse.BooleanOptionalAction)
                                      ):
                            if isinstance(module_arg, argparse.BooleanOptionalAction):
                                # Swap the "--no-" prefixed flag for the typical prefix
                                # widget_kwargs.pop('description')
                                # widget_kwargs['description'] = module_arg.option_strings[-2]
                                boolean_optional_flags[module_arg.option_strings[-2]] = long_argument
                                long_argument = module_arg.option_strings[-2]

                            # widget = boolean_widget(description=description, **widget_kwargs)  # , tooltip=help)
                            # Using the Checkbox
                            widget = boolean_widget(indent=False, **widget_kwargs, description=tooltip)
                            # widget = widgets.Box([widgets.Label(value=long_argument, tooltip=tooltip, layout=description_layout),
                            #                       widget], layout=row_layout)
                        elif isinstance(module_arg, argparse._StoreAction):
                            if choices:
                                if module_arg.default not in choices:
                                    # When there is no default provided
                                    choices_ = (module_arg.default, *choices)
                                else:
                                    choices_ = choices

                                if long_argument == '--design-method':
                                    try:
                                        bad_index = choices_.index('rosetta')
                                    except ValueError:
                                        pass
                                    else:
                                        choices_ = (*choices_[:bad_index], *choices_[bad_index + 1:])

                                    try:
                                        bad_index = choices_.index('consensus')
                                    except ValueError:
                                        pass
                                    else:
                                        choices_ = (*choices_[:bad_index], *choices_[bad_index + 1:])

                                if any(isinstance(choice, (int, float)) for choice in choices):
                                    choices_ = tuple(str(choice) for choice in choices)
                                # print(f'Found the final choices: {choices_}')
                                widget = choices_widget(**widget_kwargs, options=choices_,
                                                        layout=input_field_layout)
                            elif type_ in [None, int, float, os.path.abspath]:
                                # These are processed as strings
                                widget = process_module_arg_to_widget(module_arg, widget_kwargs)
                            elif get_class_that_defined_method(type_) == str:
                                widget = process_module_arg_to_widget(module_arg, widget_kwargs)
                                # if module_arg.nargs:
                                #     # There are currently none of these
                                #     widget = multiple_text(**widget_kwargs, placeholder=module_arg.metavar)
                                # else:
                                #     if module_arg.metavar:
                                #         placeholder = module_arg.metavar
                                #     else:
                                #         placeholder = ''
                                #     widget = single_text(**widget_kwargs, placeholder=placeholder)
                            # Custom types
                            elif type_ == flags.temp_gt0:
                                # print(widget_kwargs['value'])
                                # widget_kwargs['value'] = [str(val) for val in widget_kwargs['value']]
                                # print(widget_kwargs['value'])
                                # START Hack TagsInput
                                # Make the default value a single string
                                widget_kwargs['value'] = str(widget_kwargs['value'][0])
                                # END
                                widget = process_module_arg_to_widget(module_arg, widget_kwargs)
                                # widget = widgets.FloatsInput(**widget_kwargs)
                            else:
                                raise RuntimeError(
                                    f"Couldn't find the type '{type_}' for the argument group {module_arg}"
                                )
                        else:
                            continue
                        # DEBUG
                        # if 'pdb-code' in module_arg.option_strings[-1]:
                        #     print(module_arg, module_arg.option_strings[-1], widget_kwargs)
                        #     display(widget)
                        desc_widget = widgets.Label(value=long_argument,
                                                    description_tooltip=tooltip,
                                                    # tooltip=tooltip,
                                                    layout=description_layout)
                        # Add the module widgets to all widgets
                        # Use with [options-inputs row, ...] formating
                        widget = widgets.Box([desc_widget, widget], layout=row_layout)
                        module_widgets.append(widget)
                        # # Use with options column / inputs column formating
                        # description_widgets.append(desc_widget)
                        # module_widgets.append(widget)

                # Used for descriptions in each widget rather than an HBox description
                # max_description_length = 0
                # for widget in module_widgets:
                #     if len(widget.description) > max_description_length:
                #         max_description_length = len(widget.description)
                #         print(max_description_length)
                #         print(widget.style)
                # module_layout = {'description_width': f'{max_description_length * 6}px'}
                # print(max_description_length * 5)
                # for widget in module_widgets:
                #     widget.style = module_layout

                module_widget = widgets.Box(children=module_widgets, layout=box_layout)
                # # Use with options column / inputs column formating
                # description_column = widgets.Box(children=description_widgets, layout=description_row_layout)
                # module_column = widgets.Box(children=module_widgets, layout=row_layout)
                # module_widget = widgets.Box(children=[description_column, module_column])  # , layout=box_layout)

                # Add the module widgets to all widgets
                # all_module_widgets.append(widgets.VBox(module_widgets))
                # all_module_widgets.append(module_widget)
                all_module_widgets[module] = module_widget

# return all_module_widgets
# all_module_widgets = argparse_modules_to_ipywidgets()

# The tab holds the options for each selected module, plus required/optional program flags
module_options_tab = widgets.Tab()
module_box_layout = widgets.Layout(display='flex',
                                   flex_flow='column',
                                   align_items='stretch',
                                   border='solid', )
# Todo set a fixed width to ensure that any dependent formatting looks the same
# width='50%')
module_use_description = widgets.Label(
    value='Fill out the form to specify job arguments regarding input, module, and output parameters. Most arguments are optional')
module_box = widgets.Box([module_use_description, module_options_tab],
                         layout=module_box_layout)  # widgets.Layout(display='flex', flex_flow='column'))
input_options = widgets.Output(layout={'border': '1px solid black'})


@input_options.capture(clear_output=True)
def prepare_input_for_user_protocol(module_tags: MultipleModule, button) -> None:
    # Set up the display to show all options for each module
    # # Accordian display
    # options_accordion = widgets.Accordion(children=all_module_widgets, titles=module_tags.value)
    # display(options_accordion)

    # Set up the tab
    # NEW
    module_options_tab.children = [
                                      # all_module_widgets[flags.input_],
                                      all_module_widgets[flags.symmetry],
                                      # all_module_widgets[flags.residue_selector]] \
                                  ] \
                                  + [all_module_widgets[module_name] for module_name in module_tags.value] \
                                  + [all_module_widgets[flags.options], all_module_widgets[flags.output]]

    # module_options_tab.children = [all_module_widgets[idx] for idx in range(number_of_modules + next(extra_module_count))]
    # module_options_titles = [flags.input_, flags.symmetry, flags.residue_selector] \
    module_options_titles = [flags.symmetry] \
                            + module_tags.value \
                            + [flags.options, flags.output]
    # This doesn't work in ipywidgets < 8
    # module_options_tab.titles = module_options_titles
    # print(module_options_tab.titles)
    # This is for ipywidgets < 8
    for idx, title in enumerate(module_options_titles):
        module_options_tab.set_title(idx, title)
    # print('Titles\n\n\n')
    # print(module_options_tab.__dict__)
    display(module_box)


symmetry_query_kwargs = dict(
    description='+ query data',
    disabled=False,
    button_style='',  # 'success', 'info', 'warning', 'danger' or ''
    tooltip='When a query is requested, click to add additional info',
    icon='turn-down-left',
    layout=widgets.Layout(width='auto')
)
symmetry_query_button_widget = widgets.Button(**symmetry_query_kwargs)


class SymmetryQueryMultipleText(MultipleTextBase):
    base_widget = widgets.Dropdown
    button = symmetry_query_button_widget
    button_kwargs = symmetry_query_kwargs
    subsequent_widget = widgets.Text


symmetry_widgets = all_module_widgets[flags.symmetry]
# Set the new symmetry tags in place of the old dropdown
# print(symmetry_widgets.children)
for widget_idx, widget in enumerate(symmetry_widgets.children):
    if getattr(widget, 'children', None):
        # print(widget.children)
        desc, box = widget.children
        if desc.value == flags.query.long:
            # print(box.options)
            query_options = box.options
            break
else:
    raise RuntimeError(
        f"Couldn't find the proper symmetry flags for performing queries"
    )
symmetry_tags = SymmetryQueryMultipleText(
    value=None, options=query_options,
    layout=multitext_individual_layout,
    subsequent_kwargs=dict(
        value='', layout=multitext_individual_layout)
)

# new_desc = widgets.Label(value=desc.value, layout=desc.layout, description_tooltip=desc.description_tooltip)
# widget.children = swidgets.Box([desc, symmetry_tags.box], layout=row_layout)
symmetry_widgets.children = symmetry_widgets.children[:widget_idx] \
                            + (widgets.Box([desc, symmetry_tags.box], layout=row_layout),) \
                            + symmetry_widgets.children[widget_idx + 1:]


# symmetry_widgets.children[widget_idx] = widgets.Box([desc, symmetry_tags.box], layout=row_layout)


def parse_gui_input(module_tags: MultipleModule) -> None:
    """Formats arguments from user GUI into parsable command-line like arguments"""
    # print(module_tags.value)

    if len(module_tags.value) > 1:
        valid_arguments = [['symdesign', 'protocol', '--modules', *module_tags.value]]
    else:
        valid_arguments = [['symdesign', *module_tags.value]]

    def add_valid_arguments(arguments: Iterable[widgets.Widget]) -> None:
        """From each widget with program options, add the flag and arguments considering the flag and argument type"""
        for argument in arguments:
            if getattr(argument, 'children', None):
                # Recurse
                add_valid_arguments(argument.children)
                continue

            try:
                arg_value = getattr(argument, 'value')
            except AttributeError:
                if not isinstance(argument, widgets.Button):
                    print(
                        f"For flag '{flag.value}' found an argument without a '.value' attribute that is unrecognized "
                        f"and wasn't parsed: {argument}")
                continue

            if arg_value is None:
                if isinstance(argument, widgets.Dropdown):
                    continue
                else:
                    print(f"Found argument with None value. {argument}")
            else:  # arg_value is not None:
                if arg_value == '':
                    continue
                elif isinstance(arg_value, bool):
                    if arg_value:
                        argument = [flag.value]
                    else:
                        if flag.value in boolean_optional_flags:
                            # print(f'{flag.value} is {argument.value}. It should be --no-')
                            # argument = [flags.make_no_argument.format(flag.value)]
                            argument = [boolean_optional_flags[flag.value]]
                        else:
                            # print(f'{flag.value} is {argument.value}. It shouldn't be --no-')
                            continue
                elif isinstance(arg_value, list):
                    argument = [flag.value, *arg_value]
                else:
                    argument = [flag.value, arg_value]

                valid_arguments.append(argument)

    for module_widget_flag_argument in module_options_tab.children:
        for flag_argument in module_widget_flag_argument.children:
            if getattr(flag_argument, 'children', None):
                # flag, arguments = flag_argument.children
                # START Hack TagsInput
                flag, *arguments = flag_argument.children
                add_valid_arguments(arguments)
                # END

    sys.argv = []
    for arg in valid_arguments:
        sys.argv.extend(arg)

    # Special case for symmetric queries
    if flags.query.long in sys.argv:
        query_index = sys.argv.index(flags.query.long)
        add_indices = [query_index]
        for index in range(query_index, len(sys.argv)):
            if sys.argv[index] == flags.query.long:
                continue
            elif '-' in sys.argv[index]:
                break
            else:
                add_indices.append(index)

        sys.argv = ['symdesign', 'symmetry'] + [sys.argv[idx] for idx in add_indices]
        if flags.nanohedra.long in sys.argv:
            sys.argv += [flags.nanohedra.long]

    # # Debugging
    # sys.argv.append('--help')
    print('For running on the command line, use the equivalent command:'
          f'\n"python {subprocess.list2cmdline(sys.argv)}"\n'
          "Ensure any special POSIX characters for filter comparisons such as "
          "'>','<', or '=' are escaped by quotes\n")


job_out = widgets.Output(layout={'border': '1px solid black'})


@job_out.capture(clear_output=True)
def run_app(module_tags: MultipleModule, button) -> None:
    parse_gui_input(module_tags)
    try:
        app()
    except SystemExit:
        return
    except (SymDesignException, StructureException) as error:
        print(error)


# All modules
# flags.available_modules
# notebook_allowed_modules = [
#     flags.nanohedra,
#     flags.analysis,
#     flags.check_clashes,
#     flags.cluster_poses,
#     flags.design,
#     flags.expand_asu,
#     flags.generate_fragments,
#     flags.helix_bending,
#     flags.protocol,
#     flags.rename_chains,
#     flags.select_designs,
#     flags.select_poses,
#     flags.select_designs,
# ]
submit_protocol = 'Process Input Options'
module_help = widgets.Label(
    value="Please input the module(s) you would like to include with this job, "
          f"then click '{submit_protocol}'")
submit_protocol_button = widgets.Button(
    description=submit_protocol,
    disabled=False,
    button_style='success',  # 'success', 'info', 'warning', 'danger' or ''
    tooltip='To enter options for the specified job(s), click this button',
    icon='turn-down-left',
    layout=widgets.Layout(width='auto')
)
# Button to run the specified input
run_protocol_button = widgets.Button(
    description='Run protocol',
    disabled=False,
    button_style='success',  # 'success', 'info', 'warning', 'danger' or ''
    tooltip='To perform the protocols specified, click this button',
    icon='turn-down-left'
)
