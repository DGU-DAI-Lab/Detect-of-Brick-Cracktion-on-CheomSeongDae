#!/bin/bash

is_pip_installed() {
    if echo "$(python3 -m pip show $1)" | grep -q "Name: $1"
    then
        echo true
    else
        echo false
    fi
}

check_python3_installed() {
    if ! command -v python3
    then
        echo ""
        echo "FATAL: Python 3.x Not found."
        echo "you must install python3."
        echo "you can download python from https://www.python.org/"
        echo ""
        exit 1
    fi
}

install_virtualenv() {
    if [[ ! -d "env" ]]
    then
        if [ is_pip_installed virtualenv == "false" ]
        then
            python3 -m pip install virtualenv
        fi
        virtualenv env --python=3.7
    fi
}

install_dependencies() {
    for module in "numpy opencv-python tensorflow"
    do
        if [ "$(is_pip_installed $module)" == "false" ]
        then
            pip install $module
        fi
    done
}

main_wrapper() {
    source env/bin/activate
    main
    deactivate
}

main() {
    cd "src"
    python -m main
}

check_python3_installed
install_virtualenv
install_dependencies
main_wrapper