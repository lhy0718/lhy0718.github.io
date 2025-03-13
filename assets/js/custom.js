document.addEventListener("DOMContentLoaded", function () {
  article_imgs = $(".page__content img")
  article_imgs.each(function () {
    $(this).on("click", function () {
      width_percent = ($(this).width() / $(this).parent().width()) * 100
      if (width_percent === 100) {
        $(this).css("width", "50%")
      } else {
        $(this).css("width", "100%")
      }
    })
  })

  url = window.location.href
  if (url.includes("tags") || url.includes("categories") || url.includes("posts")) {

    $(".taxonomy__section").hide()

    window.onload = function () {
      selected_id = window.location.href.split("#")[1]
      document.getElementById(selected_id).scrollIntoView();
      $(".taxonomy__section").hide()
      $(".taxonomy__section").each(function () {
          if ($(this).attr("id") === selected_id) {
            $(this).show()
          }
        }
      )
    }

    window.onhashchange = function () {
      selected_id = window.location.href.split("#")[1]
      document.getElementById(selected_id).scrollIntoView();
      $(".taxonomy__section").hide()
      $(".taxonomy__section").each(function () {
          if ($(this).attr("id") === selected_id) {
            $(this).show()
          }
        }
      )
    }

    for (i = 0; i < $(".taxonomy__index li").length; i++) {
      $(".taxonomy__index li")[i].onclick = function () {
        selected = $(this).find("a").attr("href").split("#")[1]
        $(".taxonomy__section").hide()
        $(".taxonomy__section").each(function () {
          if ($(this).attr("id") === selected) {
            $(this).show()
          }
        })
      }
    }
  }
})