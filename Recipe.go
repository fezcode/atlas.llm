//go:build gobake
package bake_recipe

import (
	"fmt"
	"github.com/fezcode/gobake"
)

func Run(bake *gobake.Engine) error {
	if err := bake.LoadRecipeInfo("recipe.piml"); err != nil {
		return err
	}

	bake.Task("build", "Builds the binary", func(ctx *gobake.Context) error {
		ctx.Log("Building %s v%s...", bake.Info.Name, bake.Info.Version)

		err := ctx.Mkdir("build")
		if err != nil {
			return err
		}

		ldflags := fmt.Sprintf("-X main.Version=%s", bake.Info.Version)
        
        output := "build/" + bake.Info.Name + ".exe"

		err = ctx.Run("go", "build", "-ldflags", ldflags, "-o", output, ".")
		if err != nil {
            return err
        }
        
        ctx.Log("Build successful: %s", output)
        return nil
	})

	return nil
}
