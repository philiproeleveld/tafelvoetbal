//$(document).ready(function(){

function buildDropDown(values) {
    let contents = []
    for (let name of values) {
    name = name.replace(/\b\w/g, l => l.toUpperCase())
    contents.push('<input type="button" class="dropdown-item" type="button" value="' + name + '"/>')
    }
    $('.menuItems').append(contents.join(""))

    //Hide the row that shows no items were found
    $('.empty').hide()
}

$('.form-control').on('input', function() {
    console.log($(this).siblings())
    let dropdown_items = $(this).siblings()[0].children
    let input_text = $(this)[0].value.trim().toLowerCase()
    filter(input_text, dropdown_items)
})

//For every word entered by the user, check if the symbol starts with that word
//If it does show the symbol, else hide it
function filter(word, items) {
    let length = items.length // lengte van usernames
    let collection = []
    let hidden = 0

    // Loop over all items in the items array
    for (let i = 0; i < length; i++) {
        // Show if there is some sort of match
        if (items[i].value.toLowerCase().startsWith(word)) {
            let word = items[i].value

            let selected_values = $('.hidden-form')
            let selected = false

            for (let j = 0; j < selected_values.length; j++) {
                if (selected_values[j].value == word) {
                    selected = true
                    console.log(word)
                }
            }

            if (selected == true) {
                $(items[i]).hide()
                hidden++
            } else {
                $(items[i]).show()
            }
        }
        else {
            $(items[i]).hide()
            hidden++
        }
    }

    //If all items are hidden, show the empty view
    if (hidden === length) {
    $('.empty').show()
    }
    else {
    $('.empty').hide()
    }
}

//If the user clicks on any item, set the title of the button as the text of the item
$('.menuItems').on('click', '.dropdown-item', function(){

    // Set title of button
    let dropdown_sibling = $(this).parent().parent().siblings(".dropdown-toggle")
    dropdown_sibling.text($(this)[0].value)
    dropdown_sibling.dropdown('toggle')

    // Also set the value of hidden form attribute to the selected value
    let post_sibling = $(this).parent().parent().siblings(".hidden-form")
    post_sibling.val($(this)[0].value)
})

buildDropDown(names)

//})

